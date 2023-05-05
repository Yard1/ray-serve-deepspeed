import torch
from transformers import PreTrainedModel

from ._base import LLMInitializer


class SingleDeviceInitializer(LLMInitializer):
    def __init__(
        self,
        device: torch.device,
        world_size: int,
        dtype: torch.dtype = torch.float16,
        **from_pretrained_kwargs
    ):
        super().__init__(
            device=device, world_size=world_size, dtype=dtype, **from_pretrained_kwargs
        )

    def get_model_from_pretrained_kwargs(self):
        return dict(torch_dtype=self.dtype, **self.kwargs)

    def postprocess_model(self, model: "PreTrainedModel") -> "PreTrainedModel":
        return model.to(device=self.device)
