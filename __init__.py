import asyncio
import os
from typing import List, Literal, Optional, Union

import pandas as pd
import ray
import ray.util
import yaml
from fastapi import FastAPI
from pydantic import BaseModel


class Prompt(BaseModel):
    prompt: str


class Framework(BaseModel):
    pass


class DeepSpeed(Framework):
    use_kernel: bool = False
    max_tokens: int = 4096


class DeviceMap(Framework):
    device_map: str = "auto"


class LLM(BaseModel):
    name: str
    mode: Union[DeviceMap, DeepSpeed]
    batch_size: int = 1
    max_new_tokens: int = 256
    mirror_bucket_uri: Optional[str] = None
    dtype: str = "float16"


class Scaling(BaseModel):
    num_worker_groups: int = 2
    num_gpus_per_worker_group: int = 2
    num_cpus_per_worker: int = 4


class Args(BaseModel):
    # bucket_uri: str = "s3://large-dl-models-mirror/models--anyscale--opt-66b-resharded/main/"
    # name: str = "facebook/opt-66b"
    # hf_home: str = "/nvme/cache"
    name: str
    hf_home: str
