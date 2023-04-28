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

from initializers import get_initializer_cls_by_name
from models import LLM, Prompt
from pipelines import get_pipeline_cls_by_name
from utils import initialize_node, timeit

WARMUP_PROMPT = "Write a short story."


@timeit
def init_model(
    llm_config: LLM,
    world_size: int,
    local_rank: int,
    batch_size: Optional[int] = None,
):
    """Initialize the model"""
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
    print(initializer)

    model, tokenizer = initializer.load(llm_config.name)

    print(model)

    pipeline_name = llm_config.pipeline_cls
    pipeline = get_pipeline_cls_by_name(pipeline_name)(model=model, tokenizer=tokenizer)

    # Warmup
    # For DS w/ kernel inject, first batch the model gets MUST be of maximum batch size,
    # otherwise subsequent batches with more entries than the first batch
    # will raise CUDA errors if use_kernel=True.
    batch_size = batch_size or 1
    max_new_tokens = llm_config.generation_kwargs.get("max_new_tokens", 256)
    resp1 = generate(
        [WARMUP_PROMPT] * batch_size, pipeline, max_new_tokens=max_new_tokens
    )
    assert len(resp1) == batch_size
    generate([WARMUP_PROMPT], pipeline, max_new_tokens=max_new_tokens)

    print("Model succesfully initialized!")

    return pipeline


@timeit
def generate(prompts: List[Prompt], pipeline, **generate_kwargs) -> List[str]:
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
        hf_home: Optional[str] = None,
        num_cpus_per_worker: int = 1,
    ):
        """Initialize model for inference"""
        os.environ["OMP_NUM_THREADS"] = str(num_cpus_per_worker)

        initialize_node(
            model_name=self.llm_config.name,
            bucket_uri=self.llm_config.mirror_bucket_uri,
            hf_home=hf_home,
        )
        self.generator = init_model(
            self.llm_config,
            self.world_size,
            local_rank,
            batch_size=self.llm_config.batch_size,
        )

    def generate(self, data: List[Prompt], column: str, **kwargs) -> List[str]:
        return generate(data, self.generator, **kwargs)


class LLMPredictor(Predictor):
    def __init__(self, checkpoint: Checkpoint, scaling_config: ScalingConfig) -> None:
        self.checkpoint = checkpoint
        self.scaling_config = scaling_config
        self.init_worker_group(scaling_config)

    def init_worker_group(self, scaling_config: ScalingConfig):
        """Create the worker group.

        Each worker in the group communicates with other workers through the
        torch distributed backend. The worker group is inelastic (a failure of
        one worker will destroy the entire group). Each worker in the group
        recieves the same input data and outputs the same generated text.
        """
        config = self.checkpoint.to_dict()["config"]
        llm_config = config.model_config

        # Start a placement group for the workers.
        self.pg = scaling_config.as_placement_group_factory().to_placement_group()
        prediction_worker_cls = PredictionWorker.options(
            num_cpus=scaling_config.num_cpus_per_worker,
            num_gpus=scaling_config.num_gpus_per_worker,
            resources=scaling_config.additional_resources_per_worker,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=self.pg, placement_group_capture_child_tasks=True
            ),
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

        print(self.prediction_workers, local_ranks)

        # Initialize model on each worker.
        ray.get(
            [
                worker.init_model.remote(
                    local_rank,
                    hf_home=config.hf_home,
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
