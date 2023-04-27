import logging
import os
import shutil
import subprocess
import sys
import time
from functools import wraps
from pathlib import Path
from typing import List, Optional

import torch
import torch.backends.cuda
from filelock import FileLock

from huggingface_utils import download_model
from initializers import get_initializer_cls_by_name
from models import LLM
from pipelines import get_pipeline_cls_by_name
from utils import timeit

WARMUP_PROMPT = "Write a short story."
logger = logging.getLogger(__name__)


def timeit(func):
    @wraps(func)
    def inner(*args, **kwargs):
        start_time = time.monotonic()
        ret = func(*args, **kwargs)
        time_taken = time.monotonic() - start_time
        print(f"{func} took {time_taken} s to complete", file=sys.stderr)
        return ret

    return inner


def initialize_node(
    model_name: Optional[str] = None,
    bucket_uri: Optional[str] = None,
    hf_home: Optional[str] = "/nvme/cache",
):
    if hf_home:
        os.environ["HF_HOME"] = hf_home

    # Timeout in 10 minutes
    lock = FileLock("/home/ray/default/nodeinit.lock", timeout=600)
    with lock:
        shutil.rmtree("/home/ray/.cache/torch", ignore_errors=True)
        os.makedirs("/home/ray/.cache/torch/kernels", exist_ok=True)
        if Path("/nvme/.done").exists():
            print("Skipping node initialization...")
        else:
            print("Executing node initialization...")
            _initialize_node(
                bucket_uri=bucket_uri,
                model_name=model_name,
            )
            subprocess.run("touch /nvme/.done", shell=True, check=True)


def _initialize_node(
    model_name: Optional[str] = None,
    bucket_uri: Optional[str] = None,
):
    # Mount nvme
    print("Mounting nvme")
    subprocess.run(
        'drive_name="${1:-/dev/nvme1n1}"; mount_path="${2:-/nvme}"; set -x; sudo file -s "$drive_name"; sudo apt install xfsprogs -y; sudo mkfs -t xfs "$drive_name"; sudo mkdir "$mount_path" && sudo mount "$drive_name" "$mount_path" && sudo chown -R ray "$mount_path"',
        shell=True,
    )

    if model_name and bucket_uri:
        download_model(model_name, bucket_uri)
        print("Done downloading the model")


@timeit
def init_model(
    llm_config: LLM,
    world_size: int,
    local_rank: int,
    batch_size: Optional[int] = None,
):
    """Initialize the model"""
    # Lazy import so that the new cache location is used
    torch.backends.cuda.matmul.allow_tf32 = True
    dtype = getattr(torch, llm_config.dtype)
    device = torch.device(f"cuda:{local_rank}")

    initializer_name = llm_config.mode
    if not isinstance(initializer_name, str):
        initializer_name = initializer_name.__class__.__name__
    initializer = get_initializer_cls_by_name(initializer_name)(
        device=device, world_size=world_size, dtype=dtype, **llm_config.mode.dict()
    )
    model, tokenizer = initializer.load(llm_config.name)

    pipeline_name = llm_config.pipeline_cls
    pipeline = get_pipeline_cls_by_name(pipeline_name)(model=model, tokenizer=tokenizer)

    # Warmup
    # The first batch the model gets MUST be of maximum batch size,
    # otherwise subsequent batches with more entries than the first batch
    # will raise CUDA errors if use_kernel=True.
    batch_size = batch_size or 1
    resp1 = generate([WARMUP_PROMPT] * batch_size, pipeline, max_new_tokens=256)
    assert len(resp1) == batch_size
    generate([WARMUP_PROMPT], pipeline, max_new_tokens=256)

    return pipeline


@timeit
def generate(input_sentences: List[str], pipeline, **generate_kwargs) -> List[str]:
    """Generate predictions using a Pipeline"""
    outputs = pipeline(
        input_sentences,
        **generate_kwargs,
    )
    return outputs
