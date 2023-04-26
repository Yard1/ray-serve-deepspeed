import argparse
import os
from typing import List

import pandas as pd
import ray
import ray.util
from ray.air import Checkpoint, ScalingConfig
from ray.air.util.torch_dist import (
    TorchDistributedWorker,
    init_torch_dist_process_group,
)
from ray.train.predictor import Predictor
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from deepspeed_utils import generate, init_model
from utils import initialize_node


@ray.remote
class PredictionWorker(TorchDistributedWorker):
    """A PredictionWorker is a Ray remote actor that runs a single shard of a DeepSpeed job.

    Multiple PredictionWorkers of the same WorkerGroup will form a PyTorch DDP process
    group and work together under the orchestration of DeepSpeed.
    """

    def __init__(self, config: argparse.Namespace, world_size: int):
        self.config = config
        self.world_size = world_size

    def init_model(self, local_rank: int):
        """Initialize model for inference"""
        os.environ["OMP_NUM_THREADS"] = str(self.config.num_cpus_per_worker)

        initialize_node(
            model_name=self.config.name,
            bucket_uri=self.config.bucket_uri,
            hf_home=self.config.hf_home,
        )
        self.generator = init_model(
            self.config, self.world_size, local_rank, batch_size=self.config.batch_size
        )

    def generate(self, data: pd.DataFrame, column: str, **kwargs) -> List[str]:
        return generate(list(data[column]), self.generator, **kwargs)


class DeepSpeedPredictor(Predictor):
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
            prediction_worker_cls.remote(config, scaling_config.num_workers)
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
                worker.init_model.remote(local_rank)
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
