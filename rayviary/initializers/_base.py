import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


class LLMInitializer(ABC):
    def __init__(
        self,
        device: torch.device,
        world_size: int,
        dtype: torch.dtype = torch.float16,
        **kwargs,
    ):
        self.device = device
        self.world_size = world_size
        self.dtype = dtype
        self.kwargs = kwargs

    @abstractmethod
    def get_model_from_pretrained_kwargs(self) -> Dict[str, Any]:
        pass

    def load(self, model_name) -> Tuple["PreTrainedModel", "PreTrainedTokenizer"]:
        model = self.load_model(model_name)
        tokenizer = self.load_tokenizer(model_name)
        return self.postprocess_model(model), self.postprocess_tokenizer(tokenizer)

    def load_model(self, model_name: str) -> "PreTrainedModel":
        from transformers.utils.hub import TRANSFORMERS_CACHE

        path = os.path.expanduser(
            os.path.join(TRANSFORMERS_CACHE, f"models--{model_name.replace('/', '--')}")
        )
        if os.path.exists(path):
            with open(os.path.join(path, "refs", "main"), "r") as f:
                snapshot_hash = f.read().strip()
            model_name = os.path.join(path, "snapshots", snapshot_hash)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, **self.get_model_from_pretrained_kwargs()
        )
        model.eval()
        return model

    def load_tokenizer(self, tokenizer_name: str) -> "PreTrainedTokenizer":
        return AutoTokenizer.from_pretrained(tokenizer_name, padding_side="left")

    def postprocess_model(self, model: "PreTrainedModel") -> "PreTrainedModel":
        return model

    def postprocess_tokenizer(
        self, tokenizer: "PreTrainedTokenizer"
    ) -> "PreTrainedTokenizer":
        return tokenizer
