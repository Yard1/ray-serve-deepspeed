from typing import List, Optional, Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from ._base import BasePipeline

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
        stopping_tokens: List[Union[int, str]] = None,
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
            stopping_tokens=stopping_tokens,
            **kwargs,
        )

    def preprocess(self, prompts: List[str], **generate_kwargs):
        prompt_text = self._construct_prompts(
            prompts,
        )
        instruction_text = self._construct_prompts(prompts, prompt_format="")
        if not self.tokenizer.pad_token or self.tokenizer.pad_token_id < 0:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        inputs = self.tokenizer(prompt_text, return_tensors="pt", padding=True)
        return {"inputs": inputs, "instruction_text": instruction_text}

    def forward(self, model_inputs, **generate_kwargs):
        inputs = model_inputs["inputs"]
        instruction_text = model_inputs["instruction_text"]
        for t in inputs:
            if torch.is_tensor(inputs[t]):
                inputs[t] = inputs[t].to(self.model.device)

        generate_kwargs = {**inputs, **dict(stopping_criteria=self.stopping_criteria), **generate_kwargs}
        generated_sequence = self.model.generate(
            **generate_kwargs
        )
        return {
            "generated_sequence": generated_sequence,
            "instruction_text": instruction_text,
        }

    def postprocess(self, model_outputs, **generate_kwargs) -> List[str]:
        tokens = model_outputs["generated_sequence"]
        instruction_text = model_outputs["instruction_text"]
        decoded = []
        for token_unwrapped in tokens:
            decoded.append(
                self.tokenizer.decode(token_unwrapped, skip_special_tokens=True)
            )
        return [
            response[response.find(instruction_text) + len(instruction_text) :]
            for response, instruction_text in zip(decoded, instruction_text)
        ]
