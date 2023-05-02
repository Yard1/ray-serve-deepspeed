from abc import ABC, abstractmethod
from collections import UserDict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union

import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

from models import Prompt

from .utils import get_special_token_id

if TYPE_CHECKING:
    from ..initializers._base import LLMInitializer


class BasePipeline(ABC):
    """Stripped down version of Transformers pipeline."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        prompt_format: Optional[str] = None,
        device: Optional[Union[str, int, torch.device]] = None,
        stopping_tokens: List[Union[int, str]] = None,
        **kwargs,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_format: str = prompt_format or ""
        self.stopping_tokens = self._get_stopping_tokens(tokenizer, stopping_tokens)
        self.stopping_criteria = self._get_stopping_criteria(self.stopping_tokens)
        self.kwargs = kwargs

        if device is None:
            # `accelerate` device map
            hf_device_map = getattr(self.model, "hf_device_map", None)
            if hf_device_map is not None:
                # Take the first device used by `accelerate`.
                device = next(iter(hf_device_map.values()))
            else:
                device = model.device

        if isinstance(device, torch.device):
            self.device = device
        elif isinstance(device, str):
            self.device = torch.device(device)
        elif device < 0:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{device}")

    @classmethod
    def from_initializer(
        cls,
        initializer: "LLMInitializer",
        model_name: str,
        prompt_format: Optional[str] = None,
        device: Optional[Union[str, int, torch.device]] = None,
        stopping_tokens: List[Union[int, str]] = None,
        **kwargs,
    ) -> "BasePipeline":
        model, tokenizer = initializer.load(model_name)
        return cls(
            model,
            tokenizer,
            prompt_format=prompt_format,
            device=device,
            stopping_tokens=stopping_tokens,
            **kwargs,
        )

    @property
    def _default_stopping_tokens(self) -> List[int]:
        return []

    def _get_stopping_criteria(
        self, stopping_tokens: List[int]
    ) -> "StoppingCriteriaList":
        class StopOnTokens(StoppingCriteria):
            def __call__(
                self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
            ) -> bool:
                stop_ids = (
                    stopping_tokens
                    if stopping_tokens is not None
                    else [50278, 50279, 50277, 1, 0]
                )
                for stop_id in stop_ids:
                    if input_ids[0][-1] == stop_id:
                        return True
                return False

        return StoppingCriteriaList([StopOnTokens()])

    def _construct_prompt(self, prompt: Union[str, Prompt], prompt_format: str) -> str:
        if isinstance(prompt, Prompt):
            if prompt.use_prompt_format and prompt_format:
                return prompt_format.format(instruction=prompt.prompt)
            else:
                return prompt.prompt
        return prompt_format.format(instruction=prompt) if prompt_format else prompt

    def _construct_prompts(
        self,
        prompts: Union[str, Prompt, List[str], List[Prompt]],
        prompt_format: Optional[str] = None,
    ) -> List[str]:
        if not isinstance(prompts, list):
            prompts = [prompts]
        return [
            self._construct_prompt(
                prompt,
                prompt_format=prompt_format
                if prompt_format is not None
                else self.prompt_format,
            )
            for prompt in prompts
        ]

    @abstractmethod
    def preprocess(self, prompts: List[str], **generate_kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward(self, model_inputs, **generate_kwargs):
        raise NotImplementedError

    def postprocess(self, model_outputs, **generate_kwargs) -> List[str]:
        return model_outputs

    def _set_default_forward_params(self, forward_params: dict) -> None:
        forward_params.setdefault("do_sample", True)
        forward_params.setdefault("top_p", 0.92)
        forward_params.setdefault("top_k", 0)

    def _get_stopping_tokens(
        self, tokenizer: PreTrainedTokenizer, stopping_tokens: List[Union[str, int]]
    ) -> Set[int]:
        if not stopping_tokens:
            return None
        return {
            get_special_token_id(tokenizer, key) if isinstance(key, str) else key
            for key in stopping_tokens
        }

    @torch.inference_mode()
    def __call__(self, inputs: List[Union[str, Prompt]], **kwargs) -> List[str]:
        (
            preprocess_params,
            forward_params,
            postprocess_params,
        ) = self._sanitize_parameters(**kwargs)
        self._set_default_forward_params(forward_params)
        model_inputs = self.preprocess(inputs, **preprocess_params)
        model_inputs = self._ensure_tensor_on_device(model_inputs, device=self.device)
        model_outputs = self.forward(model_inputs, **forward_params)
        model_outputs = self._ensure_tensor_on_device(
            model_outputs, device=torch.device("cpu")
        )

        outputs = self.postprocess(model_outputs, **postprocess_params)
        return outputs

    def ensure_tensor_on_device(self, **inputs):
        """
        Ensure PyTorch tensors are on the specified device.

        Args:
            inputs (keyword arguments that should be `torch.Tensor`, the rest is ignored):
                The tensors to place on `self.device`.
            Recursive on lists **only**.

        Return:
            `Dict[str, torch.Tensor]`: The same as `inputs` but on the proper device.
        """
        return self._ensure_tensor_on_device(inputs, self.device)

    def _ensure_tensor_on_device(self, inputs, device: torch.device):
        from transformers.utils import ModelOutput

        if isinstance(inputs, ModelOutput):
            return ModelOutput(
                {
                    name: self._ensure_tensor_on_device(tensor, device)
                    for name, tensor in inputs.items()
                }
            )
        elif isinstance(inputs, dict):
            return {
                name: self._ensure_tensor_on_device(tensor, device)
                for name, tensor in inputs.items()
            }
        elif isinstance(inputs, UserDict):
            return UserDict(
                {
                    name: self._ensure_tensor_on_device(tensor, device)
                    for name, tensor in inputs.items()
                }
            )
        elif isinstance(inputs, list):
            return [self._ensure_tensor_on_device(item, device) for item in inputs]
        elif isinstance(inputs, tuple):
            return tuple(
                [self._ensure_tensor_on_device(item, device) for item in inputs]
            )
        elif isinstance(inputs, torch.Tensor):
            if device == torch.device("cpu") and inputs.dtype in {
                torch.float16,
                torch.bfloat16,
            }:
                inputs = inputs.float()
            return inputs.to(device)
        else:
            return inputs

    def _sanitize_parameters(
        self, return_full_text: bool = None, **generate_kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        preprocess_params = {}

        forward_params = generate_kwargs
        postprocess_params = {}

        if return_full_text is not None:
            postprocess_params["return_full_text"] = return_full_text

        return preprocess_params, forward_params, postprocess_params
