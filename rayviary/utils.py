import os
import shutil
import subprocess
import time
from functools import wraps
from pathlib import Path
from typing import Optional

from filelock import FileLock

from rayviary.huggingface_utils import download_model
from rayviary.logger import get_logger

logger = get_logger(__name__)


def timeit(func):
    @wraps(func)
    def inner(*args, **kwargs):
        start_time = time.monotonic()
        ret = func(*args, **kwargs)
        time_taken = time.monotonic() - start_time
        logger.info(f"{func} took {time_taken} s to complete")
        return ret

    return inner


def initialize_node(
    model_name: Optional[str] = None,
    bucket_uri: Optional[str] = None,
):
    # Timeout in 10 minutes
    with FileLock("/home/ray/default/nodeinit.lock", timeout=600):
        shutil.rmtree("/home/ray/.cache/torch", ignore_errors=True)
        os.makedirs("/home/ray/.cache/torch/kernels", exist_ok=True)
        if Path("/nvme/.done").exists():
            logger.info("Skipping nvme initialization...")
        else:
            logger.info("Executing nvme initialization...")
            _initialize_nvme()
            subprocess.run("touch /nvme/.done", shell=True, check=True)
    if model_name and bucket_uri:
        with FileLock(
            f"/home/ray/default/{model_name.replace('/', '--')}.lock", timeout=1200
        ):
            download_model(model_name, bucket_uri)
            logger.info("Done downloading the model!")


def _initialize_nvme():
    # Mount nvme
    logger.info("Mounting nvme...")
    subprocess.run(
        'drive_name="${1:-/dev/nvme1n1}"; mount_path="${2:-/nvme}"; set -x; sudo file -s "$drive_name"; sudo apt install xfsprogs -y; sudo mkfs -t xfs "$drive_name"; sudo mkdir "$mount_path" && sudo mount "$drive_name" "$mount_path" && sudo chown -R ray "$mount_path"',
        shell=True,
    )
