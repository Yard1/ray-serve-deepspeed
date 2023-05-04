import gc
import os
from typing import List, Optional

import pandas as pd
import ray
import ray.util
import torch
import torch.backends.cuda
from ray.air import Checkpoint, ScalingConfig
from ray.air.util.torch_dist import (
    TorchDistributedWorker,
    init_torch_dist_process_group,
)
from ray.train.predictor import Predictor
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from rayviary.initializers import get_initializer_cls_by_name
from rayviary.logger import get_logger
from rayviary.models import LLM, Args, Prompt, Response
from rayviary.pipelines import get_pipeline_cls_by_name
from rayviary.utils import initialize_node, timeit

WARMUP_PROMPT = "Write a short story."

initialize_node_remote = ray.remote(initialize_node)

logger = get_logger(__name__)


@timeit
def init_model(
    llm_config: LLM,
    world_size: int,
    local_rank: int,
    max_batch_size: Optional[int] = None,
):
    """Initialize the model"""
    logger.info(f"Initializing model {llm_config.name}...")

    # Lazy import so that the new cache location is used
    torch.backends.cuda.matmul.allow_tf32 = True
    dtype = getattr(torch, llm_config.dtype)
    device = torch.device(f"cuda:{local_rank}")

    initializer_name = llm_config.mode
    if not isinstance(initializer_name, str):
        initializer_name = initializer_name.type
    initializer = get_initializer_cls_by_name(initializer_name)(
        device=device,
        world_size=world_size,
        dtype=dtype,
        **llm_config.mode.dict(exclude={"type"}),
    )

    pipeline_name = llm_config.pipeline_cls
    pipeline = get_pipeline_cls_by_name(pipeline_name).from_initializer(
        initializer,
        llm_config.name,
        prompt_format=llm_config.prompt_format,
        **llm_config.from_pretrained_kwargs,
    )

    # Warmup
    # For DS w/ kernel inject, first batch the model gets MUST be of maximum batch size,
    # otherwise subsequent batches with more entries than the first batch
    # will raise CUDA errors if use_kernel=True.
    batch_size = max_batch_size or 1
    logger.info(f"Model {llm_config.name} is warming up...")
    resp1 = generate(
        [WARMUP_PROMPT] * batch_size, pipeline, **llm_config.generation_kwargs
    )
    assert len(resp1) == batch_size
    assert all(len(x.generated_text) > 1 for x in resp1)
    resp2 = generate([WARMUP_PROMPT], pipeline, **llm_config.generation_kwargs)
    assert len(resp2) == batch_size
    assert all(len(x.generated_text) > 1 for x in resp2)

    logger.info(f"Model {llm_config.name} succesfully initialized!")

    return pipeline


@timeit
def generate(prompts: List[Prompt], pipeline, **generate_kwargs) -> List[Response]:
    """Generate predictions using a Pipeline"""
    outputs = pipeline(
        prompts,
        **generate_kwargs,
    )
    return outputs


@ray.remote
class PredictionWorker(TorchDistributedWorker):
    """A PredictionWorker is a Ray remote actor that runs a single shard of a DeepSpeed job.

    Multiple PredictionWorkers of the same WorkerGroup will form a PyTorch DDP process
    group and work together under the orchestration of DeepSpeed.
    """

    def __init__(self, llm_config: LLM, world_size: int):
        self.llm_config = llm_config
        self.world_size = world_size

    def init_model(
        self,
        local_rank: int,
        num_cpus_per_worker: int = 1,
    ):
        get_logger(__name__, rank=int(os.environ["RANK"]), force=True)

        """Initialize model for inference"""
        os.environ["OMP_NUM_THREADS"] = str(num_cpus_per_worker)

        self.generator = init_model(
            self.llm_config,
            self.world_size,
            local_rank,
            max_batch_size=self.llm_config.max_batch_size,
        )

    def generate(self, data: List[Prompt], **kwargs) -> List[str]:
        return generate(data, self.generator, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}:{self.llm_config.name}"


class LLMPredictor(Predictor):
    def __init__(self, checkpoint: Checkpoint, scaling_config: ScalingConfig) -> None:
        self.checkpoint = checkpoint
        self.scaling_config = scaling_config
        self.prediction_workers = None
        self.init_worker_group(scaling_config)

    def init_worker_group(self, scaling_config: ScalingConfig):
        """Create the worker group.

        Each worker in the group communicates with other workers through the
        torch distributed backend. The worker group is inelastic (a failure of
        one worker will destroy the entire group). Each worker in the group
        recieves the same input data and outputs the same generated text.
        """
        self.prediction_workers = None
        gc.collect()

        config: Args = self.checkpoint.to_dict()["config"]
        llm_config = config.model_config

        # Start a placement group for the workers.
        self.pg = scaling_config.as_placement_group_factory().to_placement_group()
        scaling_options = dict(
            num_cpus=scaling_config.num_cpus_per_worker,
            num_gpus=scaling_config.num_gpus_per_worker,
            resources=scaling_config.additional_resources_per_worker,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=self.pg, placement_group_capture_child_tasks=True
            ),
        )
        runtime_env = {"env_vars": {"HF_HOME": config.hf_home}}
        prediction_worker_cls = PredictionWorker.options(
            **scaling_options, runtime_env=runtime_env
        )
        initialize_node_remote_pg = initialize_node_remote.options(
            **scaling_options, runtime_env=runtime_env
        )
        ray.get(
            [
                initialize_node_remote_pg.remote(
                    llm_config.name, llm_config.mirror_bucket_uri
                )
                for i in range(scaling_config.num_workers)
            ]
        )

        # Create the prediction workers.
        self.prediction_workers = [
            prediction_worker_cls.remote(llm_config, scaling_config.num_workers)
            for i in range(scaling_config.num_workers)
        ]

        # Initialize torch distributed process group for the workers.
        local_ranks = init_torch_dist_process_group(
            self.prediction_workers, backend="nccl"
        )

        # Initialize model on each worker.
        ray.get(
            [
                worker.init_model.remote(
                    local_rank,
                    num_cpus_per_worker=scaling_config.num_cpus_per_worker,
                )
                for worker, local_rank in zip(self.prediction_workers, local_ranks)
            ]
        )

    def _predict_pandas(
        self,
        data: pd.DataFrame,
        input_column: str = "prompt",
        output_column: str = "output",
        **kwargs,
    ) -> pd.DataFrame:
        data_ref = ray.put(data)
        prediction = ray.get(
            [
                worker.generate.remote(data_ref, column=input_column, **kwargs)
                for worker in self.prediction_workers
            ]
        )[0]

        return pd.DataFrame(prediction, columns=[output_column])

    @classmethod
    def from_checkpoint(cls, checkpoint: Checkpoint, **kwargs) -> "Predictor":
        return cls(checkpoint=checkpoint, **kwargs)
