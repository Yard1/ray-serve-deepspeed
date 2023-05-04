import deepspeed
import torch
from transformers import AutoModelForCausalLM

from ..logger import get_logger
from ._base import LLMInitializer

logger = get_logger(__name__)


# TODO: Allow deepspeed kwargs
class DeepSpeedInitializer(LLMInitializer):
    def __init__(
        self,
        device: torch.device,
        world_size: int,
        dtype: torch.dtype = torch.float16,
        max_tokens: int = 1024,
        use_kernel: bool = False,
        injection_policy=None,
        **from_pretrained_kwargs,
    ):
        super().__init__(
            device=device, world_size=world_size, dtype=dtype, **from_pretrained_kwargs
        )
        self.max_tokens = max_tokens
        self.use_kernel = use_kernel
        # TODO: Allow conversion from strings (need to do dynamic imports)
        self.injection_policy = injection_policy

    def get_model_from_pretrained_kwargs(self):
        return dict(low_cpu_mem_usage=True, torch_dtype=self.dtype, **self.kwargs)

    def postprocess_model(
        self, model: "AutoModelForCausalLM"
    ) -> "AutoModelForCausalLM":
        from transformers import GPTNeoXForCausalLM, LlamaForCausalLM

        if not self.injection_policy and not self.use_kernel:
            if isinstance(model, GPTNeoXForCausalLM):
                from transformers import GPTNeoXLayer

                injection_policy = {
                    GPTNeoXLayer: ("attention.dense", "mlp.dense_4h_to_h")
                }
            elif isinstance(model, LlamaForCausalLM):
                from transformers.models.llama.modeling_llama import LlamaDecoderLayer

                injection_policy = {
                    LlamaDecoderLayer: ("self_attn.o_proj", "mlp.down_proj")
                }
        else:
            injection_policy = self.injection_policy

        model = deepspeed.init_inference(
            model,
            dtype=self.dtype,
            mp_size=self.world_size,
            replace_with_kernel_inject=self.use_kernel,
            injection_policy=injection_policy,
            max_tokens=self.max_tokens,
        )

        # Add attributes for compatibility with the pipeline
        model.use_kernel = self.use_kernel
        model.device = self.device
        model = model.to(self.device)
        logger.info(f"DeepSpeed model: {model}")
        return model
