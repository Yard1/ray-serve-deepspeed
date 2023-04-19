from ray import serve
from fastapi import FastAPI
from pydantic import BaseModel
from ray import serve
import os
from dataclasses import dataclass
import asyncio

from pathlib import Path


import os
from argparse import ArgumentParser

import pandas as pd
import ray
import ray.util
from ray.air import Checkpoint, ScalingConfig
from ray.train.batch_predictor import BatchPredictor

import subprocess


from deepspeed_predictor import DeepSpeedPredictor, PredictionWorker, initialize_node

from dataclasses import dataclass

import os
from collections import defaultdict
from typing import List, Optional, Tuple

import pandas as pd
import ray
import ray.util
from ray.air import ScalingConfig
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

import yaml

import ray

app = FastAPI()


class Prompt(BaseModel):
    prompt: str


class Args(BaseModel):
    # bucket_uri: str = "s3://large-dl-models-mirror/models--anyscale--opt-66b-resharded/main/"
    # name: str = "facebook/opt-66b"
    # hf_home: str = "/nvme/cache"
    # checkpoint_path: str = "/nvme/model"
    bucket_uri: str
    name: str
    hf_home: str
    checkpoint_path: str
    batch_size: int = 1
    ds_inference: bool = True
    use_kernel: bool = False
    # use_meta_tensor: bool = False
    num_worker_groups: int = 1
    num_gpus_per_worker_group: int = 2
    reshard_checkpoint_path: Optional[str] = None
    # use_cache: bool = True

    max_new_tokens: int = 256
    max_tokens: int = 1024
    dtype: str = "float16"
    save_mp_checkpoint_path: Optional[str] = None


raw_args = os.getenv("APPLICATION_ARGS")
dict_args = {
      "bucket_uri": "",
      "name": "diegi97/dolly-v2-12b-sharded-bf16",
      "hf_home": "/nvme/cache",
      "checkpoint_path": "/nvme/model"
}
if raw_args:
    assert raw_args is not None, "APPLICATION_ARGS env var must be set"
    print("Received args", raw_args)
    dict_args = yaml.load(raw_args, Loader=yaml.SafeLoader)
    print("Received args dict", dict_args)
args = Args.parse_obj(dict_args)


ray.init(
    address="auto",
    runtime_env={
        "pip": [
            "numpy==1.23",
            "protobuf==3.20.0",
            "transformers==4.28.1",
            "accelerate==0.18.0",
            "deepspeed==0.9.0",
        ],
        "env_vars": {
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            #"HF_HOME": args.hf_home
        }
    }
)

@serve.deployment(
    route_prefix="/", num_replicas=args.num_worker_groups,
)
@serve.ingress(app)
class DeepspeedApp(DeepSpeedPredictor):
    def __init__(self) -> None:
        self.args = args

        scaling_config = ScalingConfig(
            use_gpu=True,
            num_workers=args.num_gpus_per_worker_group,
            trainer_resources={"CPU": 0},
        )

        self.checkpoint = Checkpoint.from_dict({"config": self.args})
        self.scaling_config = scaling_config
        self.init_worker_group(scaling_config)

    @app.post("/")
    async def generate_text(self, prompt: Prompt):
        return await self.generate_text_batch(prompt)

    @serve.batch(max_batch_size=args.batch_size, batch_wait_timeout_s=1)
    async def generate_text_batch(self, prompts: List[Prompt]):
        """Generate text from the given prompts in batch """

        print("Received prompts", prompts)
        input_column = "predict"
        #  Wrap in pandas
        data = pd.DataFrame(
            [prompt.prompt for prompt in prompts], columns=[input_column]
        )
        data_ref = ray.put(data)
        prediction = (
            await asyncio.gather(
                *[
                    worker.generate.remote(
                        data_ref,
                        column=input_column,
                        #do_sample=True,
                        #temperature=0.9,
                        max_new_tokens=args.max_new_tokens,
                    )
                    for worker in self.prediction_workers
                ]
            )
        )[0]
        print("Predictions", prediction)
        return prediction[: len(prompts)]

entrypoint = DeepspeedApp.bind()

# The following block will be executed if the script is run by Python directly
if __name__ == "__main__":
    serve.run(entrypoint)
