from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Extra, Field, root_validator, validator
from typing_extensions import Annotated


class BaseModelExtended(BaseModel):
    @classmethod
    def parse_yaml(cls, file):
        dict_args = yaml.load(file, Loader=yaml.SafeLoader)
        return cls.parse_obj(dict_args)

    def yaml(
        self,
        *,
        stream=None,
        include=None,
        exclude=None,
        by_alias: bool = False,
        skip_defaults: Union[bool, None] = None,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        **kwargs,
    ):
        return yaml.dump(
            self.dict(
                include=include,
                exclude=exclude,
                by_alias=by_alias,
                skip_defaults=skip_defaults,
                exclude_unset=exclude_unset,
                exclude_defaults=exclude_defaults,
                exclude_none=exclude_none,
            ),
            stream=stream,
            **kwargs,
        )


class ComputedPropertyMixin:
    # Replace with pydantic.computed_field once it's available
    @classmethod
    def get_properties(cls):
        return [prop for prop in dir(cls) if isinstance(getattr(cls, prop), property)]

    def dict(self, *args, **kwargs):
        self.__dict__.update(
            {prop: getattr(self, prop) for prop in self.get_properties()}
        )
        return super().dict(*args, **kwargs)

    def json(
        self,
        *args,
        **kwargs,
    ) -> str:
        self.__dict__.update(
            {prop: getattr(self, prop) for prop in self.get_properties()}
        )

        return super().json(*args, **kwargs)


class Prompt(BaseModelExtended):
    prompt: str
    use_prompt_format: bool = True

    def __str__(self) -> str:
        return self.prompt


class Response(ComputedPropertyMixin, BaseModelExtended):
    generated_text: str
    num_generated_tokens: Optional[int] = None
    num_generated_tokens_batch: Optional[int] = None
    preprocessing_time: Optional[float] = None
    generation_time: Optional[float] = None
    postprocessing_time: Optional[float] = None

    @property
    def total_time(self) -> Optional[float]:
        try:
            return (
                self.preprocessing_time
                + self.generation_time
                + self.postprocessing_time
            )
        except Exception:
            return None

    @property
    def total_time_per_token(self) -> Optional[float]:
        try:
            return self.total_time / self.num_generated_tokens
        except Exception:
            return None

    @property
    def generation_time_per_token(self) -> Optional[float]:
        try:
            return self.generation_time / self.num_generated_tokens
        except Exception:
            return None

    @property
    def total_time_per_token_batch(self) -> Optional[float]:
        try:
            return self.total_time / self.num_generated_tokens_batch
        except Exception:
            return None

    @property
    def generation_time_per_token_batch(self) -> Optional[float]:
        try:
            return self.generation_time / self.num_generated_tokens_batch
        except Exception:
            return None

    def __str__(self) -> str:
        return self.generated_text


class Framework(BaseModelExtended, extra=Extra.forbid):
    type: str

    @root_validator(pre=True)
    def set_type(cls, values):
        values["type"] = cls.__name__
        return values


class DeepSpeed(Framework):
    type: Literal["DeepSpeed"]
    use_kernel: bool = False
    max_tokens: int = 4096


class DeviceMap(Framework):
    type: Literal["DeviceMap"]
    device_map: str = "auto"


class LLM(BaseModelExtended):
    name: str
    mode: Annotated[Union[DeviceMap, DeepSpeed], Field(discriminator="type")]
    pipeline_cls: str
    prompt_format: Optional[str] = None
    max_batch_size: int = 1
    batch_wait_timeout_s: int = 1
    dtype: str = "float16"
    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": 256,
        "do_sample": True,
        "top_p": 0.92,
        "top_k": 0,
    }
    from_pretrained_kwargs: Dict[str, Any] = {}
    stopping_tokens: Optional[List[Union[str, int, List[Union[str, int]]]]] = None
    mirror_bucket_uri: Optional[str] = None

    @validator("prompt_format")
    def check_prompt_format(cls, value):
        if value:
            assert (
                "{instruction}" in value
            ), "prompt_format must be None, empty string or string containing '{instruction}'"
        return value

    @validator("stopping_tokens")
    def check_stopping_tokens(cls, value):
        def try_int(x):
            if isinstance(x, list):
                return [try_int(y) for y in x]
            try:
                return int(x)
            except Exception:
                return x

        if value:
            value = try_int(value)
        return value


class Scaling(BaseModelExtended):
    num_workers: int
    num_gpus_per_worker: float = 1
    num_cpus_per_worker: float = 1
    placement_strategy: str = "PACK"
    resources_per_worker: Optional[Dict[str, float]] = None


class Args(BaseModelExtended):
    model_config: LLM
    scaling_config: Scaling
    hf_home: Optional[str] = None
