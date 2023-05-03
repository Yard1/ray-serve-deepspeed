import time
from typing import List, Optional, Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from ._base import BasePipeline
from .utils import remove_dangling_stop_tokens

SYSTEM_PROMPT = """<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
"""


PROMPT_FOR_GENERATION_FORMAT = SYSTEM_PROMPT + "<|USER|>{instruction}<|ASSISTANT|>"


class StableLMPipeline(BasePipeline):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        prompt_format: Optional[str] = None,
        device: Optional[Union[str, int, torch.device]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            prompt_format=(
                prompt_format
                if prompt_format is not None
                else PROMPT_FOR_GENERATION_FORMAT
            ),
            device=device,
            **kwargs,
        )

    @property
    def _default_stopping_tokens(self) -> List[int]:
        return [50278, 50279, 50277, 1, 0]

    def preprocess(self, prompts: List[str], **generate_kwargs):
        st = time.monotonic()
        prompt_text = self._construct_prompts(
            prompts,
        )
        instruction_text = self._construct_prompts(prompts, prompt_format="")
        if not self.tokenizer.pad_token or self.tokenizer.pad_token_id < 0:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        inputs = self.tokenizer(
            prompt_text, return_tensors="pt", padding=True, **generate_kwargs
        ).to(self.model.device)
        if not generate_kwargs.get("return_token_type_ids", True):
            inputs.pop("token_type_ids", None)
        et = time.monotonic() - st
        return {
            "inputs": inputs,
            "instruction_text": instruction_text,
            "prompt_text": prompt_text,
            "preprocessing_time": et,
        }

    def forward(self, model_inputs, **generate_kwargs):
        st = time.monotonic()
        inputs = model_inputs["inputs"]
        instruction_text = model_inputs["instruction_text"]
        prompt_text = model_inputs["prompt_text"]
        preprocessing_time = model_inputs["preprocessing_time"]

        generate_kwargs = {
            **inputs,
            **generate_kwargs,
        }
        generated_sequence = self.model.generate(**generate_kwargs)
        et = time.monotonic() - st
        return {
            "inputs": inputs,
            "generated_sequence": generated_sequence,
            "instruction_text": instruction_text,
            "prompt_text": prompt_text,
            "preprocessing_time": preprocessing_time,
            "generation_time": et,
        }

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
        (
            preprocess_params,
            forward_params,
            postprocess_params,
        ) = super()._sanitize_parameters(
            return_full_text,
            return_tensors,
            return_text,
            return_type,
            clean_up_tokenization_spaces,
            prefix,
            handle_long_generation,
            stop_sequence,
            return_token_type_ids,
            stopping_tokens,
            **generate_kwargs,
        )
        postprocess_params["stopping_tokens"] = stopping_tokens
        return preprocess_params, forward_params, postprocess_params

    def postprocess(self, model_outputs, **generate_kwargs) -> List[str]:
        st = time.monotonic()
        tokens = model_outputs["generated_sequence"]
        input_ids = model_outputs["inputs"]["input_ids"]
        decoded = []
        stopping_tokens = generate_kwargs.pop("stopping_tokens", None)
        stop_ids = (
            stopping_tokens
            if stopping_tokens is not None
            else self._default_stopping_tokens
        )
        stop_ids = [
            torch.LongTensor([stop_id] if not isinstance(stop_id, list) else stop_id)
            for stop_id in stop_ids
        ]
        eos_token_ids = self.tokenizer.all_special_ids + [0]
        for token_unwrapped, inputs_unwrapped in zip(tokens, input_ids):
            tokens = token_unwrapped[len(inputs_unwrapped) :]
            tokens = remove_dangling_stop_tokens(tokens, stop_ids, eos_token_ids)
            text = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
            decoded.append(text)
        et = time.monotonic() - st
        return decoded
