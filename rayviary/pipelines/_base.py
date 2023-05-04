import warnings
from abc import ABC, abstractmethod
from collections import UserDict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union

import torch
from transformers import (
    InfNanRemoveLogitsProcessor,
    LogitsProcessorList,
    PreTrainedModel,
    PreTrainedTokenizer,
    StoppingCriteriaList,
)
from transformers.pipelines.text_generation import ReturnType

from ..models import Prompt, Response
from .processors import StopOnTokensLogitsProcessor
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
        **kwargs,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_format: str = prompt_format or ""
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
        **kwargs,
    ) -> "BasePipeline":
        model, tokenizer = initializer.load(model_name)
        return cls(
            model,
            tokenizer,
            prompt_format=prompt_format,
            device=device,
            **kwargs,
        )

    @property
    def _default_stopping_tokens(self) -> List[int]:
        return []

    def _get_stopping_criteria(
        self, generate_kwargs: Dict[str, Any]
    ) -> StoppingCriteriaList:
        return StoppingCriteriaList([])

    def _get_logits_processors(
        self, generate_kwargs: Dict[str, Any]
    ) -> LogitsProcessorList:
        if getattr(self.model, "use_kernel", False):
            return LogitsProcessorList([])
        lst = []
        stopping_tokens = self._default_stopping_tokens
        if generate_kwargs.get("stopping_tokens", None) is not None:
            stopping_tokens = self._get_stopping_tokens(
                self.tokenizer, generate_kwargs.pop("stopping_tokens")
            )
        if stopping_tokens:
            lst.append(
                StopOnTokensLogitsProcessor(
                    stopping_tokens, self.tokenizer.eos_token_id
                )
            )

        # if getattr(self.model, "use_kernel", False):
        #     lst.append(InfNanRemoveLogitsProcessor())

        return LogitsProcessorList(lst)

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

    def postprocess(self, model_outputs, **generate_kwargs) -> List[Response]:
        return model_outputs

    def _get_stopping_tokens(
        self, tokenizer: PreTrainedTokenizer, stopping_tokens: List[Union[str, int]]
    ) -> Set[int]:
        if not stopping_tokens:
            return None
        return [
            get_special_token_id(tokenizer, key) if isinstance(key, str) else key
            for key in stopping_tokens
        ]

    @torch.inference_mode()
    def __call__(self, inputs: List[Union[str, Prompt]], **kwargs) -> List[Response]:
        (
            preprocess_params,
            forward_params,
            postprocess_params,
        ) = self._sanitize_parameters(**kwargs)
        model_inputs = self.preprocess(inputs, **preprocess_params)
        model_inputs = self._ensure_tensor_on_device(model_inputs, device=self.device)
        model_outputs = self.forward(model_inputs, **forward_params)
        model_outputs = self._ensure_tensor_on_device(
            model_outputs, device=torch.device("cpu")
        )

        outputs = self.postprocess(model_outputs, **postprocess_params)
        return [
            Response(generated_text=text) if isinstance(text, str) else text
            for text in outputs
        ]

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

    def _add_default_generate_kwargs(
        self, generate_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        stopping_criteria = self._get_stopping_criteria(generate_kwargs)
        if stopping_criteria:
            if generate_kwargs.get("stopping_criteria", None):
                generate_kwargs["stopping_criteria"].extend(stopping_criteria)
                generate_kwargs["stopping_criteria"] = StoppingCriteriaList(
                    generate_kwargs["stopping_criteria"]
                )
            else:
                generate_kwargs["stopping_criteria"] = StoppingCriteriaList(
                    stopping_criteria
                )

        logits_processor = self._get_logits_processors(generate_kwargs)
        if logits_processor:
            if generate_kwargs.get("logits_processor", None):
                generate_kwargs["logits_processor"].extend(logits_processor)
                generate_kwargs["logits_processor"] = LogitsProcessorList(
                    generate_kwargs["logits_processor"]
                )
            else:
                generate_kwargs["logits_processor"] = LogitsProcessorList(
                    logits_processor
                )

        generate_kwargs.pop("stopping_tokens", None)
        return generate_kwargs

    def _sanitize_parameters(
        self,
        return_full_text=None,
        return_tensors=None,
        return_text=None,
        return_type=None,
        clean_up_tokenization_spaces=None,
        prefix=None,
        handle_long_generation=None,
        stop_sequence=None,
        return_token_type_ids=None,
        stopping_tokens=None,
        **generate_kwargs,
    ):
        preprocess_params = {}
        if prefix is not None:
            preprocess_params["prefix"] = prefix
        if return_token_type_ids is not None:
            preprocess_params["return_token_type_ids"] = return_token_type_ids
        if prefix:
            prefix_inputs = self.tokenizer(
                prefix, padding=False, add_special_tokens=False, return_tensors="pt"
            )
            prefix_length = prefix_inputs["input_ids"].shape[-1]

            if "max_new_tokens" in generate_kwargs:
                pass
            elif "max_length" in generate_kwargs:
                generate_kwargs["max_length"] += prefix_length
            else:
                generate_kwargs["max_length"] = (
                    self.model.config.max_length + prefix_length
                )

            if "min_length" in generate_kwargs:
                generate_kwargs["min_length"] += prefix_length
        if handle_long_generation is not None:
            if handle_long_generation not in {"hole"}:
                raise ValueError(
                    f"{handle_long_generation} is not a valid value for `handle_long_generation` parameter expected"
                    " [None, 'hole']"
                )
            preprocess_params["handle_long_generation"] = handle_long_generation

        if stopping_tokens is not None:
            generate_kwargs["stopping_tokens"] = stopping_tokens
        forward_params = self._add_default_generate_kwargs(generate_kwargs)

        postprocess_params = {}
        if return_full_text is not None and return_type is None:
            if return_text is not None:
                raise ValueError(
                    "`return_text` is mutually exclusive with `return_full_text`"
                )
            if return_tensors is not None:
                raise ValueError(
                    "`return_full_text` is mutually exclusive with `return_tensors`"
                )
            return_type = (
                ReturnType.FULL_TEXT if return_full_text else ReturnType.NEW_TEXT
            )
        if return_tensors is not None and return_type is None:
            if return_text is not None:
                raise ValueError(
                    "`return_text` is mutually exclusive with `return_tensors`"
                )
            return_type = ReturnType.TENSORS
        if return_type is not None:
            postprocess_params["return_type"] = return_type
        if clean_up_tokenization_spaces is not None:
            postprocess_params[
                "clean_up_tokenization_spaces"
            ] = clean_up_tokenization_spaces

        if stop_sequence is not None:
            stop_sequence_ids = self.tokenizer.encode(
                stop_sequence, add_special_tokens=False
            )
            if len(stop_sequence_ids) > 1:
                warnings.warn(
                    "Stopping on a multiple token sequence is not yet supported on transformers. The first token of"
                    " the stop sequence will be used as the stop sequence string in the interim."
                )
            generate_kwargs["eos_token_id"] = stop_sequence_ids[0]

        return preprocess_params, forward_params, postprocess_params
