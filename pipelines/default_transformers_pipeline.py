import torch
from transformers import pipeline

from ._base import BasePipeline


class DefaultTransformersPipeline(BasePipeline):
    def __init__(
        self,
        model,
        tokenizer,
        prompt_format=None,
        device=None,
        stopping_tokens=None,
        **kwargs
    ) -> None:
        super().__init__(
            model, tokenizer, prompt_format, device, stopping_tokens, **kwargs
        )

        self.pipeline = None

    def _get_transformers_pipeline(self, **kwargs):
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
    def __call__(self, inputs, **kwargs):
        if not self.pipeline:
            self.pipeline = self._get_transformers_pipeline()
        default_kwargs = dict(stopping_criteria=self.stopping_criteria)
        inputs = [str(input) for input in inputs]
        return self.pipeline(inputs, **{**default_kwargs, **kwargs})

    @classmethod
    def from_initializer(
        cls,
        initializer,
        model_name,
        prompt_format=None,
        device=None,
        stopping_tokens=None,
        **kwargs
    ):
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

    def preprocess(self, prompts, **generate_kwargs):
        pass

    def forward(self, model_inputs, **generate_kwargs):
        pass
