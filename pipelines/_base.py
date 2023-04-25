from abc import ABC, abstractmethod
from collections import UserDict

import torch


class BasePipeline(ABC):
    """Stripped down version of Transformers pipeline."""

    def __init__(self, model, tokenizer, device=None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

        if device is not None and not (isinstance(device, int) and device < 0):
            self.model.to(device)

        if device is None:
            # `accelerate` device map
            hf_device_map = getattr(self.model, "hf_device_map", None)
            if hf_device_map is not None:
                # Take the first device used by `accelerate`.
                device = next(iter(hf_device_map.values()))
            else:
                device = -1

        if isinstance(device, torch.device):
            self.device = device
        elif isinstance(device, str):
            self.device = torch.device(device)
        elif device < 0:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{device}")

    @abstractmethod
    def preprocess(self, instruction_text, **generate_kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward(self, model_inputs, **generate_kwargs):
        raise NotImplementedError

    def postprocess(self, model_outputs, **generate_kwargs):
        return model_outputs

    def _set_default_forward_params(self, forward_params):
        forward_params.setdefault("do_sample", True)
        forward_params.setdefault("top_p", 0.92)
        forward_params.setdefault("top_k", 0)

    def __call__(self, inputs, **kwargs):
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

    def _ensure_tensor_on_device(self, inputs, device):
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

    def _sanitize_parameters(self, return_full_text: bool = None, **generate_kwargs):
        preprocess_params = {}

        forward_params = generate_kwargs
        postprocess_params = {}

        if return_full_text is not None:
            postprocess_params["return_full_text"] = return_full_text

        return preprocess_params, forward_params, postprocess_params
