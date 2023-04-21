import logging
import os
import subprocess
import sys
import time
from functools import wraps
from pathlib import Path
from typing import Optional

from filelock import FileLock

from huggingface_utils import download_model

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
    hf_home: str = "/nvme/cache",
):
    os.environ["HF_HOME"] = hf_home

    # Timeout in 10 minutes
    lock = FileLock("/home/ray/default/nodeinit.lock", timeout=600)
    with lock:
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

    if bucket_uri:
        download_model(model_name, bucket_uri)
    print("Done downloading the model")
