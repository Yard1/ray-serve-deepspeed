from typing import TYPE_CHECKING, List, Optional, Union

import torch
from transformers import Pipeline as TransformersPipeline
from transformers import PreTrainedModel, PreTrainedTokenizer, pipeline

from models import Prompt

from ._base import BasePipeline

if TYPE_CHECKING:
    from ..initializers._base import LLMInitializer


class DefaultTransformersPipeline(BasePipeline):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        prompt_format: Optional[str] = None,
        device: Optional[Union[str, int, torch.device]] = None,
        **kwargs
    ) -> None:
        super().__init__(model, tokenizer, prompt_format, device, **kwargs)

        self.pipeline = None

    def _get_transformers_pipeline(self, **kwargs) -> TransformersPipeline:
        default_kwargs = dict(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=None,
        )
        transformers_pipe = pipeline(**{**default_kwargs, **self.kwargs, **kwargs})
        transformers_pipe.device = self.device
        return transformers_pipe

    @torch.inference_mode()
    def __call__(self, inputs: List[Union[str, Prompt]], **kwargs) -> List[str]:
        if not self.pipeline:
            self.pipeline = self._get_transformers_pipeline()
        kwargs = self._add_default_generate_kwargs(kwargs)
        inputs = [str(input) for input in inputs]
        return self.pipeline(inputs, **kwargs)

    @classmethod
    def from_initializer(
        cls,
        initializer: "LLMInitializer",
        model_name: str,
        prompt_format: Optional[str] = None,
        device: Optional[Union[str, int, torch.device]] = None,
        stopping_tokens: List[Union[int, str]] = None,
        **kwargs
    ) -> "DefaultTransformersPipeline":
        default_kwargs = dict(
            model=model_name,
            device=None,
        )
        transformers_pipe = pipeline(
            **{**default_kwargs, **kwargs},
            model_kwargs=initializer.get_model_from_pretrained_kwargs()
        )
        transformers_pipe.model = initializer.postprocess_model(transformers_pipe.model)
        pipe = cls(
            model=transformers_pipe.model,
            tokenizer=transformers_pipe.tokenizer,
            prompt_format=prompt_format,
            device=device,
            stopping_tokens=stopping_tokens,
            **kwargs
        )
        pipe.pipeline = transformers_pipe
        transformers_pipe.device = pipe.device
        return pipe

    def preprocess(self, prompts: List[str], **generate_kwargs):
        pass

    def forward(self, model_inputs, **generate_kwargs):
        pass
