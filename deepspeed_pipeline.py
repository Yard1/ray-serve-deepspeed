# DeepSpeed inference related utils.
# Modeled after https://github.com/microsoft/DeepSpeedExamples/tree/master/inference/huggingface/text-generation

import gc
import io
import json
import math
import os
from pathlib import Path
from typing import List

import deepspeed
import torch
from deepspeed.runtime.utils import see_memory_usage
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline


class DSPipeline:
    """
    Example helper class for comprehending DeepSpeed Meta Tensors, meant to mimic HF pipelines.
    The DSPipeline can run with and without meta tensors.
    """

    def __init__(
        self,
        model_name,
        dtype=torch.float16,
        is_meta=True,
        device=-1,
        repo_root=None,
    ):
        self.model_name = model_name
        self.dtype = dtype

        if isinstance(device, torch.device):
            self.device = device
        elif isinstance(device, str):
            self.device = torch.device(device)
        elif device < 0:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if is_meta:
            # When meta tensors enabled, use checkpoints
            self.config = AutoConfig.from_pretrained(self.model_name)
            self.checkpoints_json = self._generate_json(repo_root)

            with deepspeed.OnDevice(dtype=dtype, device="meta"):
                self.model = AutoModelForCausalLM.from_config(self.config)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=dtype, low_cpu_mem_usage=True)

        self.model.eval()

    def __call__(self, inputs, **kwargs):
        input_list = [inputs] if isinstance(inputs, str) else inputs
        outputs = self.generate_outputs(input_list, **kwargs)
        return outputs

    def _generate_json(self, repo_root):
        if os.path.exists(os.path.join(repo_root, "ds_inference_config.json")):
            # Simply use the available inference config.
            return os.path.join(repo_root, "ds_inference_config.json")

        # Write a checkpoints config file in local directory.
        checkpoints_json = "checkpoints.json"

        with io.open(checkpoints_json, "w", encoding="utf-8") as f:
            file_list = [
                str(entry).split("/")[-1]
                for entry in Path(repo_root).rglob("*.[bp][it][n]")
                if entry.is_file()
            ]
            data = {
                # Hardcode bloom for now.
                # Possible choices are "bloom", "ds_model", "Megatron"
                "type": "bloom",
                "checkpoints": file_list,
                "version": 1.0
            }
            json.dump(data, f)

        return checkpoints_json

    def generate_outputs(self, inputs, **generate_kwargs):
        input_tokens = self.tokenizer.batch_encode_plus(
            inputs, return_tensors="pt", padding=True
        )
        for t in input_tokens:
            if torch.is_tensor(input_tokens[t]):
                input_tokens[t] = input_tokens[t].to(self.device)

        self.model.cuda().to(self.device)

        outputs = self.model.generate(**input_tokens, **generate_kwargs)
        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return outputs