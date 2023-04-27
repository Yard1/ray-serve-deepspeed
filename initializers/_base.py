from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMInitializer(ABC):
    def __init__(
        self,
        device: torch.device,
        world_size: int,
        dtype: torch.dtype = torch.float16,
        **kwargs
    ):
        self.device = device
        self.world_size = world_size
        self.dtype = dtype
        self.kwargs = kwargs

    def load(self, model_name) -> Tuple["AutoModelForCausalLM", "AutoTokenizer"]:
        model = self._load_model(model_name)
        tokenizer = self._load_tokenizer(model_name)
        return self._postprocess_model(model), self._postprocess_tokenizer(tokenizer)

    @abstractmethod
    def _load_model(self, model_name: str) -> "AutoModelForCausalLM":
        pass

    def _load_tokenizer(self, tokenizer_name: str) -> "AutoTokenizer":
        return AutoTokenizer.from_pretrained(tokenizer_name, padding_side="left")

    def _postprocess_model(
        self, model: "AutoModelForCausalLM"
    ) -> "AutoModelForCausalLM":
        return model

    def _postprocess_tokenizer(self, tokenizer: "AutoTokenizer") -> "AutoTokenizer":
        return tokenizer
