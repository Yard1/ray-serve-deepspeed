from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel


class BaseModelExtended(BaseModel):
    @classmethod
    def parse_yaml(cls, file):
        dict_args = yaml.load(file, Loader=yaml.SafeLoader)
        return cls.parse_obj(dict_args)


class Prompt(BaseModelExtended):
    prompt: str


class Framework(BaseModelExtended):
    pass


class DeepSpeed(Framework):
    use_kernel: bool = False
    max_tokens: int = 4096


class DeviceMap(Framework):
    device_map: str = "auto"


class LLM(BaseModelExtended):
    name: str
    mode: Union[DeviceMap, DeepSpeed]
    pipeline_cls: str
    prompt: Optional[str] = None
    batch_size: int = 1
    dtype: str = "float16"
    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": 256,
        "do_sample": True,
        "top_p": 0.92,
        "top_k": 0,
    }
    from_pretrained_kwargs: Dict[str, Any] = {}
    stopping_tokens: Optional[List[int]] = None
    mirror_bucket_uri: Optional[str] = None


class Scaling(BaseModelExtended):
    num_workers: int
    num_gpus_per_worker: int = 1
    num_cpus_per_worker: int = 1


class Args(BaseModelExtended):
    # bucket_uri: str = "s3://large-dl-models-mirror/models--anyscale--opt-66b-resharded/main/"
    # name: str = "facebook/opt-66b"
    # hf_home: str = "/nvme/cache"
    model_config: LLM
    scaling_config: Scaling
    hf_home: Optional[str] = None
