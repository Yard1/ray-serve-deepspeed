import gc
import os
import shutil
import subprocess
from collections import defaultdict
from unittest.mock import patch

import torch
from filelock import FileLock

from rayviary.logger import get_logger

logger = get_logger(__name__)


def set_transformers_cache(location: str):
    import huggingface_hub.constants
    import transformers.utils.hub

    os.makedirs(location, exist_ok=True)

    old_cache_path = os.path.join(transformers.utils.hub.hf_cache_home, "hub")
    transformers.utils.hub.hf_cache_home = os.path.expanduser(location)
    huggingface_hub.constants.hf_cache_home = transformers.utils.hub.hf_cache_home
    transformers.utils.hub.default_cache_path = os.path.join(
        transformers.utils.hub.hf_cache_home, "hub"
    )
    huggingface_hub.constants.default_cache_path = (
        transformers.utils.hub.default_cache_path
    )
    huggingface_hub.constants.default_assets_cache_path = os.path.join(
        huggingface_hub.constants.hf_cache_home, "assets"
    )

    huggingface_hub.constants.HUGGINGFACE_HUB_CACHE = os.getenv(
        "HUGGINGFACE_HUB_CACHE", huggingface_hub.constants.default_cache_path
    )
    huggingface_hub.constants.HUGGINGFACE_ASSETS_CACHE = os.getenv(
        "HUGGINGFACE_ASSETS_CACHE", huggingface_hub.constants.default_assets_cache_path
    )

    # Onetime move from the old location to the new one.
    if os.path.isdir(old_cache_path) and not os.path.isdir(
        transformers.utils.hub.default_cache_path
    ):
        shutil.move(old_cache_path, transformers.utils.hub.default_cache_path)

    transformers.utils.hub.PYTORCH_PRETRAINED_BERT_CACHE = os.getenv(
        "PYTORCH_PRETRAINED_BERT_CACHE", transformers.utils.hub.default_cache_path
    )
    transformers.utils.hub.PYTORCH_TRANSFORMERS_CACHE = os.getenv(
        "PYTORCH_TRANSFORMERS_CACHE",
        transformers.utils.hub.PYTORCH_PRETRAINED_BERT_CACHE,
    )
    transformers.utils.hub.HUGGINGFACE_HUB_CACHE = os.getenv(
        "HUGGINGFACE_HUB_CACHE", transformers.utils.hub.PYTORCH_TRANSFORMERS_CACHE
    )
    transformers.utils.hub.TRANSFORMERS_CACHE = os.getenv(
        "TRANSFORMERS_CACHE", transformers.utils.hub.HUGGINGFACE_HUB_CACHE
    )
    transformers.utils.hub.HF_MODULES_CACHE = os.getenv(
        "HF_MODULES_CACHE",
        os.path.join(transformers.utils.hub.hf_cache_home, "modules"),
    )
    transformers.utils.hub.cache_version_file = os.path.join(
        transformers.utils.hub.TRANSFORMERS_CACHE, "version.txt"
    )
    transformers.utils.hub.move_cache(
        transformers.utils.hub.TRANSFORMERS_CACHE,
        transformers.utils.hub.TRANSFORMERS_CACHE,
    )


def shard_checkpoint_contiguous(
    state_dict, max_shard_size="10GB", weights_name: str = "pytorch_model.bin"
):
    """
    Same as transformers.modeling_utils.shard_checkpoint, but shards each layer
    into its own file to mitigate https://github.com/microsoft/DeepSpeed/issues/3084.
    """
    # Lazy import so that the new cache location is used
    from transformers.modeling_utils import dtype_byte_size
    from transformers.utils.hub import convert_file_size_to_int

    max_shard_size = convert_file_size_to_int(max_shard_size)

    sharded_state_dicts = []
    current_block = {}
    current_block_size = 0
    total_size = 0

    layers = defaultdict(list)
    saved_keys = set()
    for key in state_dict:
        if key.startswith("model.decoder.layers."):
            layer_key = ".".join(key.split(".")[:4])
            layers[layer_key].append(key)

    for keys in layers.values():
        for key in keys:
            weight = state_dict[key]
            weight_size = weight.numel() * dtype_byte_size(weight.dtype)

            current_block[key] = weight
            current_block_size += weight_size
            total_size += weight_size
            saved_keys.add(key)
        sharded_state_dicts.append(current_block)
        current_block = {}
        current_block_size = 0

    for key, weight in state_dict.items():
        if key in saved_keys:
            continue
        weight_size = weight.numel() * dtype_byte_size(weight.dtype)

        # If this weight is going to tip up over the maximal size, we split.
        if current_block_size + weight_size > max_shard_size:
            sharded_state_dicts.append(current_block)
            current_block = {}
            current_block_size = 0

        current_block[key] = weight
        current_block_size += weight_size
        total_size += weight_size

    # Add the last block
    sharded_state_dicts.append(current_block)

    # If we only have one shard, we return it
    if len(sharded_state_dicts) == 1:
        return {weights_name: sharded_state_dicts[0]}, None

    # Otherwise, let's build the index
    weight_map = {}
    shards = {}
    for idx, shard in enumerate(sharded_state_dicts):
        shard_file = weights_name.replace(
            ".bin", f"-{idx+1:05d}-of-{len(sharded_state_dicts):05d}.bin"
        )
        shard_file = shard_file.replace(
            ".safetensors",
            f"-{idx + 1:05d}-of-{len(sharded_state_dicts):05d}.safetensors",
        )
        shards[shard_file] = shard
        for key in shard.keys():
            weight_map[key] = shard_file

    # Add the metadata
    metadata = {"total_size": total_size}
    index = {"metadata": metadata, "weight_map": weight_map}
    return shards, index


def reshard_checkpoint(model_name_or_path, dtype, path_to_save_in):
    """
    Loads a transformers model into CPU memory, reshards and saves it to mitigate
    https://github.com/microsoft/DeepSpeed/issues/3084.
    """
    import deepspeed
    from transformers import AutoModelForCausalLM

    with FileLock(f"{path_to_save_in}.lock"):
        # We use a done marker file so that the other ranks do not
        # go through the process again.
        done_marker = os.path.join(path_to_save_in, ".done")
        if not os.path.exists(done_marker):
            dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
            with deepspeed.OnDevice(dtype=dtype, device="cpu"):
                model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                )
            with patch(
                "transformers.modeling_utils.shard_checkpoint",
                shard_checkpoint_contiguous,
            ):
                model.save_pretrained(path_to_save_in)
            with open(done_marker, "w"):
                pass
            del model
            gc.collect()
    return path_to_save_in


def download_model(model_name: str, bucket_uri: str):
    from transformers.utils.hub import TRANSFORMERS_CACHE

    logger.info(f"Downloading {model_name} from {bucket_uri} to '{TRANSFORMERS_CACHE}'")
    path = os.path.expanduser(
        os.path.join(TRANSFORMERS_CACHE, f"models--{model_name.replace('/', '--')}")
    )
    subprocess.run(
        ["aws", "s3", "cp", "--quiet", os.path.join(bucket_uri, "hash"), "."]
    )
    with open(os.path.join(".", "hash"), "r") as f:
        f_hash = f.read().strip()
    subprocess.run(["mkdir", "-p", os.path.join(path, "snapshots", f_hash)])
    subprocess.run(["mkdir", "-p", os.path.join(path, "refs")])
    subprocess.run(
        [
            "aws",
            "s3",
            "sync",
            "--quiet",
            bucket_uri,
            os.path.join(path, "snapshots", f_hash),
        ]
    )
    with open(os.path.join(path, "refs", "main"), "w") as f:
        f.write(f_hash)
