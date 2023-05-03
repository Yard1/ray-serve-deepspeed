import asyncio
from typing import Any, Dict, List, Optional

import ray
import ray.util
from fastapi import FastAPI
from ray import serve
from ray.air import Checkpoint, ScalingConfig
from ray.experimental.state.api import list_actors
from ray.serve.batching import _BatchQueue
from ray.serve.context import get_global_client
from ray.serve.deployment import ClassNode

from models import Args, Prompt, DeepSpeed
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
    health_check_period_s=10,
    health_check_timeout_s=30,
)
class LLMDeployment(LLMPredictor):
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.args = None
        self.scaling_config = None
        self.prediction_workers = None
        if config:
            self.reconfigure(config)

    @app.post("/")
    async def generate_text(self, prompt: Prompt):
        text = await self.generate_text_batch(prompt)
        return {"generated_text": text}

    def _should_reinit_worker_group(
        self, new_args: Args, new_scaling_config: ScalingConfig
    ) -> None:
        old_scaling_config = self.scaling_config
        old_args = self.args

        if not self.prediction_workers:
            return True

        if old_scaling_config != new_scaling_config:
            return True

        if not old_args:
            return True

        if old_args.model_config.name != new_args.model_config.name:
            return True

        if old_args.model_config.mode != new_args.model_config.mode:
            return True

        if old_args.model_config.dtype != new_args.model_config.dtype:
            return True
        
        if old_args.model_config.max_batch_size != new_args.model_config.max_batch_size and isinstance(new_args.model_config.mode, DeepSpeed):
            return True

        # TODO: Allow those below
        if old_args.model_config.pipeline_cls != new_args.model_config.pipeline_cls:
            return True

        if (
            old_args.model_config.from_pretrained_kwargs
            != new_args.model_config.from_pretrained_kwargs
        ):
            return True

        if (
            old_args.model_config.stopping_tokens
            != new_args.model_config.stopping_tokens
        ):
            return True

        if old_args.model_config.prompt_format != new_args.model_config.prompt_format:
            return True

        return False

    def reconfigure(self, config: Dict[str, Any]) -> None:
        if not isinstance(config, Args):
            new_args = Args.parse_obj(config)
        else:
            new_args = config

        new_scaling_config = ScalingConfig(
            use_gpu=True,
            num_workers=new_args.scaling_config.num_workers,
            trainer_resources={"CPU": 0},
            resources_per_worker={
                "CPU": new_args.scaling_config.num_cpus_per_worker,
                "GPU": new_args.scaling_config.num_gpus_per_worker,
            },
        )

        should_reinit_worker_group = self._should_reinit_worker_group(
            new_args, new_scaling_config
        )

        self.args = new_args
        self.checkpoint = Checkpoint.from_dict({"config": self.args})
        self.scaling_config = new_scaling_config
        if should_reinit_worker_group:
            self.init_worker_group(self.scaling_config)
        asyncio.create_task(self.generate_text_batch(None))
        print("reconfigured")

    def _set_batch_queue_batch_size(self):
        # Serve abuse to enable dynamic batch sizes
        batch_queue: _BatchQueue = getattr(
            self, "__serve_batch_queue_generate_text_batch", None
        )
        if batch_queue is not None:
            batch_queue.max_batch_size = self.args.model_config.max_batch_size
            batch_queue.timeout_s = self.args.model_config.batch_wait_timeout_s

    @serve.batch(max_batch_size=1, batch_wait_timeout_s=1)
    async def generate_text_batch(self, prompts: List[Prompt]):
        """Generate text from the given prompts in batch"""
        self._set_batch_queue_batch_size()
        if not prompts or prompts[0] is None:
            return prompts
        print(f"Received {len(prompts)} prompts", prompts)
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

    # Called by Serve to check the replica's health.
    def check_health(self):
        return True
        if self.pg and self.prediction_workers:
            dead_actors = list_actors(
                filters=[
                    ("placement_group_id", "=", self.pg.id.hex()),
                    ("state", "=", "DEAD"),
                ],
                raise_on_missing_output=False,
                limit=10000,
            )
            if dead_actors:
                raise RuntimeError(
                    f"At least one prediction worker is dead. Dead workers: {dead_actors}"
                )


@serve.deployment(
    route_prefix="/",
)
@serve.ingress(app)
class RouterDeployment:
    def __init__(self, models: Dict[str, ClassNode]) -> None:
        self._models = models
        print(self._models)

    @app.post("/query/{model}")
    async def query(self, model: str, prompt: Prompt):
        model = model.replace("--", "/")
        if model == "all":
            keys = list(self._models.keys())
            models = list(self._models.values())
        else:
            keys = [model]
            models = [self._models[model]]
        prompts = await asyncio.gather(
            *(
                await asyncio.gather(
                    *[model.generate_text.remote(prompt) for model in models]
                )
            )
        )
        print(prompts)
        return {key: prompt for key, prompt in zip(keys, prompts)}

    @app.get("/models")
    async def models(self) -> List[str]:
        return list(self._models.keys())


entrypoint = RouterDeployment.bind(
    {
        "CarperAI/stable-vicuna-13b-delta": LLMDeployment.bind(),
        "lmsys/vicuna-13b-delta-v1.1": LLMDeployment.bind(),
        "stabilityai/stablelm-tuned-alpha-7b": LLMDeployment.bind(),
        "databricks/dolly-v2-12b": LLMDeployment.bind(),
    }
)

# The following block will be executed if the script is run by Python directly
if __name__ == "__main__":
    serve.run(entrypoint)
