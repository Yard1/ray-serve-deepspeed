import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ._base import LLMInitializer


class DeviceMapInitializer(LLMInitializer):
    def __init__(
        self,
        device: torch.device,
        world_size: int,
        dtype: torch.dtype = torch.float16,
        device_map: str = "auto",
        **from_pretrained_kwargs
    ):
        super().__init__(
            device=device, world_size=world_size, dtype=dtype, **from_pretrained_kwargs
        )
        self.device_map = device_map

    def get_model_from_pretrained_kwargs(self):
        return dict(torch_dtype=self.dtype, device_map=self.device_map, **self.kwargs)
