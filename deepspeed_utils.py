# Based on https://github.com/microsoft/DeepSpeedExamples/tree/master/inference/huggingface/text-generation

import argparse
from typing import List, Optional

import torch
import torch.backends.cuda

from utils import timeit

WARMUP_PROMPT = "Write a short story."


@timeit
def init_model(
    args: argparse.Namespace,
    world_size: int,
    local_rank: int,
    batch_size: Optional[int] = None,
):
    """Initialize the deepspeed model"""
    # Lazy import so that the new cache location is used
    import deepspeed
    from deepspeed.runtime.utils import see_memory_usage
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    from pipelines.dolly2_pipeline import DollyV2Pipeline
    from pipelines.stablelm_pipeline import StableLMPipeline

    torch.backends.cuda.matmul.allow_tf32 = True

    data_type = getattr(torch, args.dtype)

    if local_rank == 0:
        see_memory_usage("before init", True)

    # pipe = pipeline(
    #     model=args.name,
    #     torch_dtype=data_type,
    #     trust_remote_code=True,
    #     model_kwargs=dict(low_cpu_mem_usage=True, use_cache=True),
    # )

    if "dolly-v2" in args.name:
        pass
    elif "stablelm" in args.name:
        pass
    else:
        raise ValueError("Unsupported model")

    tokenizer = AutoTokenizer.from_pretrained(args.name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        args.name, low_cpu_mem_usage=True, torch_dtype=data_type, use_cache=True
    )
    model.eval()

    if local_rank == 0:
        see_memory_usage("after init", True)

    if args.ds_inference:
        from transformers import GPTNeoXForCausalLM, GPTNeoXLayer

        if isinstance(model, GPTNeoXForCausalLM) and not args.use_kernel:
            injection_policy = {GPTNeoXLayer: ("attention.dense", "mlp.dense_4h_to_h")}
        else:
            injection_policy = None

        model = deepspeed.init_inference(
            model,
            dtype=data_type,
            mp_size=world_size,
            replace_with_kernel_inject=args.use_kernel,
            injection_policy=injection_policy,
            max_tokens=args.max_tokens,
        )

    if local_rank == 0:
        see_memory_usage("after init_inference", True)

    device = torch.device(f"cuda:{local_rank}")
    # Add this attribute for compatibility with the pipeline
    model.device = device
    model = model.to(device)

    if "dolly-v2" in args.name:
        pipe = DollyV2Pipeline(model=model, tokenizer=tokenizer, device=device)
    elif "stablelm" in args.name:
        pipe = StableLMPipeline(model=model, tokenizer=tokenizer, device=device)

    # Warmup
    # The first batch the model gets MUST be of maximum batch size,
    # otherwise subsequent batches with more entries than the first batch
    # will raise CUDA errors if use_kernel=True.
    batch_size = batch_size or 1
    resp1 = generate([WARMUP_PROMPT] * batch_size, pipe, max_new_tokens=256)
    assert len(resp1) == batch_size
    print(generate([WARMUP_PROMPT], pipe, max_new_tokens=256))

    return pipe


@timeit
@torch.inference_mode()
def generate(input_sentences: List[str], pipe, **generate_kwargs) -> List[str]:
    """Generate predictions using a Pipeline"""
    outputs = pipe(
        input_sentences,
        **generate_kwargs,
    )
    return outputs
