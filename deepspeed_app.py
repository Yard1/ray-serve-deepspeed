import asyncio
import time
from typing import Any, Dict, List, Optional

import ray
import ray.util
from fastapi import FastAPI
from ray import serve
from ray.air import Checkpoint, ScalingConfig
from ray.serve.batching import _BatchQueue
from ray.serve.deployment import ClassNode

from models import Args, Prompt
from predictor import LLMPredictor

app = FastAPI()


# raw_args = os.getenv("APPLICATION_ARGS")
# assert raw_args is not None, "APPLICATION_ARGS env var must be set"
# print("Received args", raw_args)
# args = Args.parse_yaml(raw_args)


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
    autoscaling_config={
        "min_replicas": 1,
        "initial_replicas": 2,
        "max_replicas": 8,
    },
    ray_actor_options={"resources": {"worker_node": 0.5}},
    max_concurrent_queries=2,  # Maximum backlog for a single replica
)
class LLMDeployment(LLMPredictor):
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        if config:
            self.reconfigure(config)

    @app.post("/")
    async def generate_text(self, prompt: Prompt):
        text = await self.generate_text_batch(prompt)
        return {"generated_text": text}

    def reconfigure(self, config: Dict[str, Any]) -> None:
        if not isinstance(config, Args):
            self.args = Args.parse_obj(config)
        else:
            self.args = config

        scaling_config = ScalingConfig(
            use_gpu=True,
            num_workers=self.args.scaling_config.num_workers,
            trainer_resources={"CPU": 0},
            resources_per_worker={
                "CPU": self.args.scaling_config.num_cpus_per_worker,
                "GPU": self.args.scaling_config.num_gpus_per_worker,
            },
        )

        self.checkpoint = Checkpoint.from_dict({"config": self.args})
        self.scaling_config = scaling_config
        self.init_worker_group(scaling_config)
        asyncio.create_task(self.generate_text_batch([]))
        print("reconfigured")

    def _set_batch_queue_batch_size(self):
        # Serve abuse to enable dynamic batch sizes
        batch_queue = getattr(self, "__serve_batch_queue_generate_text_batch", None)
        if batch_queue is not None:
            batch_queue.max_batch_size = self.args.model_config.max_batch_size

    @serve.batch(max_batch_size=1, batch_wait_timeout_s=1)
    async def generate_text_batch(self, prompts: List[Prompt]):
        """Generate text from the given prompts in batch"""
        self._set_batch_queue_batch_size()
        print(f"Received {len(prompts)} prompts", prompts)
        if not prompts:
            return []
        data_ref = ray.put(prompts)
        prediction = (
            await asyncio.gather(
                *[
                    worker.generate.remote(
                        data_ref,
                        **self.args.model_config.generation_kwargs,
                    )
                    for worker in self.prediction_workers
                ]
            )
        )[0]
        print("Predictions", prediction)
        if not isinstance(prediction, list):
            return [prediction]
        return prediction[: len(prompts)]


@serve.deployment(
    route_prefix="/",
)
@serve.ingress(app)
class RouterDeployment:
    def __init__(self, models: Dict[str, ClassNode]) -> None:
        self.models = models

    @app.post("/query/{model}")
    async def query(self, model: str, prompt: Prompt):
        if model == "all":
            keys = list(self.models.keys())
            models = list(self.models.values())
        else:
            keys = [model]
            models = self.models[model]
        prompts = await asyncio.gather(
            *(
                await asyncio.gather(
                    *[model.generate_text.remote(prompt) for model in models]
                )
            )
        )
        print(prompts)
        return {key: prompt for key, prompt in zip(keys, prompts)}


entrypoint = RouterDeployment.bind(
    {"stablelm-7b": LLMDeployment.bind(), "dolly-v2": LLMDeployment.bind()}
)

# The following block will be executed if the script is run by Python directly
if __name__ == "__main__":
    serve.run(entrypoint)
