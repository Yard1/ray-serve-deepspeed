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
        **kwargs
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
            **kwargs
        )


class Prompt(BaseModelExtended):
    prompt: str


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
    batch_size: int = 1
    dtype: str = "float16"
    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": 256,
        "do_sample": True,
        "top_p": 0.92,
        "top_k": 0,
    }
    from_pretrained_kwargs: Dict[str, Any] = {}
    stopping_tokens: Optional[List[Union[str, int]]] = None
    mirror_bucket_uri: Optional[str] = None

    @validator("prompt_format")
    def check_prompt_format(cls, value):
        if value:
            assert (
                "{instruction}" in value
            ), "prompt_format must be None, empty string or string containing '{instruction}'"


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
