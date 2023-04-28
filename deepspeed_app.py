import asyncio
import os
from typing import List

import pandas as pd
import ray
import ray.util
from fastapi import FastAPI
from ray import serve
from ray.air import Checkpoint, ScalingConfig

from models import Args, Prompt
from predictor import LLMPredictor

app = FastAPI()


raw_args = os.getenv("APPLICATION_ARGS")
assert raw_args is not None, "APPLICATION_ARGS env var must be set"
print("Received args", raw_args)
args = Args.parse_yaml(raw_args)


ray.init(
    address="auto",
    runtime_env={
        "pip": [
            "numpy==1.23",
            "protobuf==3.20.0",
            "transformers==4.28.1",
            "accelerate==0.18.0",
            "deepspeed==0.9.1",
        ],
        "env_vars": {"HF_HUB_DISABLE_PROGRESS_BARS": "1"},
    },
    ignore_reinit_error=True,
)


@serve.deployment(
    route_prefix="/",
    autoscaling_config={
        "min_replicas": 1,
        "initial_replicas": 2,
        "max_replicas": 8,
    },
    ray_actor_options={"resources": {"worker_node": 0.5}},
    max_concurrent_queries=2,  # Maximum backlog for a single replica
)
@serve.ingress(app)
class DeepspeedApp(LLMPredictor):
    def __init__(self) -> None:
        self.args = args

        scaling_config = ScalingConfig(
            use_gpu=True,
            num_workers=args.scaling_config.num_workers,
            trainer_resources={"CPU": 0},
            resources_per_worker={
                "CPU": args.scaling_config.num_cpus_per_worker,
                "GPU": args.scaling_config.num_gpus_per_worker,
            },
        )

        self.checkpoint = Checkpoint.from_dict({"config": self.args})
        self.scaling_config = scaling_config
        self.init_worker_group(scaling_config)

    def _validate_args(self, args):
        pass

    @app.post("/")
    async def generate_text(self, prompt: Prompt):
        return await self.generate_text_batch(prompt)

    @serve.batch(max_batch_size=args.model_config.batch_size, batch_wait_timeout_s=1)
    async def generate_text_batch(self, prompts: List[Prompt]):
        """Generate text from the given prompts in batch"""

        print(f"Received {len(prompts)} prompts", prompts)
        data_ref = ray.put(prompts)
        prediction = (
            await asyncio.gather(
                *[
                    worker.generate.remote(
                        data_ref,
                        **args.model_config.generation_kwargs,
                    )
                    for worker in self.prediction_workers
                ]
            )
        )[0]
        print("Predictions", prediction)
        if not isinstance(prediction, list):
            return [prediction]
        return prediction[: len(prompts)]


entrypoint = DeepspeedApp.bind()

# The following block will be executed if the script is run by Python directly
if __name__ == "__main__":
    serve.run(entrypoint)
