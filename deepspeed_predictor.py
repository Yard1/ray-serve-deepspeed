import argparse
import os
import socket
from collections import defaultdict
from contextlib import closing
from datetime import timedelta
from typing import List, Optional
import subprocess
from pathlib import Path

import pandas as pd
import ray
import ray.util
import torch.distributed as dist
from ray.air import Checkpoint, ScalingConfig
from ray.train.constants import DEFAULT_NCCL_SOCKET_IFNAME
from ray.train.predictor import Predictor
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from ray.air.util.torch_dist import (
    TorchDistributedWorker,
    init_torch_dist_process_group,
)

from deepspeed_utils import generate, init_model
from huggingface_utils import reshard_checkpoint


from filelock import Timeout, FileLock


def find_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def initialize_node(bucket_uri: Optional[str]=None, path_to_save_in: str = "/nvme/model"):
    # Timeout in 10 minutes
    lock = FileLock("/home/ray/default/nodeinit.lock", timeout=600)
    with lock:
        if Path("/nvme/.done").exists():
            print("Skipping node initialization...")
            return
        else:
            print("Executing node initialization...")
            _initialize_node(bucket_uri=bucket_uri, path_to_save_in=path_to_save_in)
            subprocess.run("touch /nvme/.done", shell=True, check=True)


def _initialize_node(path_to_save_in:str, bucket_uri: Optional[str]=None):
    # Mount nvme
    print("Mounting nvme")
    subprocess.run(
        'drive_name="${1:-/dev/nvme1n1}"; mount_path="${2:-/nvme}"; set -x; sudo file -s "$drive_name"; sudo apt install xfsprogs -y; sudo mkfs -t xfs "$drive_name"; sudo mkdir "$mount_path" && sudo mount "$drive_name" "$mount_path" && sudo chown -R ray "$mount_path"',
        shell=True,
    )

    if bucket_uri:
        subprocess.run(
            ["aws", "s3", "sync", "--no-progress", bucket_uri, path_to_save_in,], check=True
        )
    print("Done downloading the model")


@ray.remote
class PredictionWorker(TorchDistributedWorker):
    def __init__(self, args: argparse.Namespace, rank: int, world_size: int):
        self.args = args
        self.rank = rank
        self.world_size = world_size

    def init_model(self, local_rank: int):
        """Initialize model for inference"""
        # Note: we have to provide the local_rank that was used to initiate
        # the DDP process group here. E.g., a PredictionWorker may be the
        # rank 0 worker of a group, but occupying gpu 7.
        self.generator = init_model(self.args, self.world_size, local_rank)

    def generate(self, data: pd.DataFrame, column: str, **kwargs) -> List[str]:
        return generate(
            list(data[column]), self.generator, self.args.batch_size, **kwargs
        )


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
        local_ranks = init_torch_dist_process_group(self.prediction_workers, backend="nccl")

        # Initialize model on each worker.
        ray.get([
            worker.init_model.remote(local_rank)
            for worker, local_rank in zip(self.prediction_workers, local_ranks)
        ])

    def _predict_pandas(
        self,
        data: pd.DataFrame,
        input_column: str = "prompt",
        output_column: str = "output",
        **kwargs
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