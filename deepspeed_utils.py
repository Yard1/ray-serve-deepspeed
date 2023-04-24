# Based on https://github.com/microsoft/DeepSpeedExamples/tree/master/inference/huggingface/text-generation

import argparse
import gc
from typing import List

from fast_pipeline import FastPipeline
from utils import timeit


@timeit
def init_model(
    args: argparse.Namespace, world_size: int, local_rank: int
) -> FastPipeline:
    """Initialize the deepspeed model"""
    # Lazy import so that the new cache location is used
    import deepspeed
    import torch
    from deepspeed.runtime.utils import see_memory_usage
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    # from deepspeed_pipeline import DSPipeline

    data_type = getattr(torch, args.dtype)

    if local_rank == 0:
        see_memory_usage("before init", True)

    # pipe = pipeline(
    #     model=args.name,
    #     torch_dtype=data_type,
    #     trust_remote_code=True,
    #     model_kwargs=dict(low_cpu_mem_usage=True, use_cache=True),
    # )
    # pipe = DSPipeline(
    #     model_name=args.name,
    #     dtype=data_type,
    #     is_meta=False,
    #     device=local_rank,
    #     #repo_root=args.repo_root,
    # )

    tokenizer = AutoTokenizer.from_pretrained(args.name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        args.name, low_cpu_mem_usage=True, torch_dtype=data_type, use_cache=True
    )

    if local_rank == 0:
        see_memory_usage("after init", True)

    ds_kwargs = dict()

    gc.collect()

    from transformers import GPTNeoXForCausalLM, GPTNeoXLayer

    if args.ds_inference:
        if isinstance(model, GPTNeoXForCausalLM):
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
            save_mp_checkpoint_path=args.save_mp_checkpoint_path,
            **ds_kwargs,
        )

    if local_rank == 0:
        see_memory_usage("after init_inference", True)

    device = torch.device(f"cuda:{local_rank}")
    model.cuda().to(device)
    # Add this attribute for compatibility with the pipeline
    model.device = device
    pipe = FastPipeline(model=model, tokenizer=tokenizer, device=device)

    # Warmup
    generate(["Test"], pipe)

    return pipe


@timeit
def generate(
    input_sentences: List[str], pipe: FastPipeline, **generate_kwargs
) -> List[str]:
    """Generate predictions using a Pipeline"""
    generate_kwargs.setdefault("do_sample", True)
    generate_kwargs.setdefault("top_p", 0.92)
    generate_kwargs.setdefault("top_k", 0)
    outputs = pipe(
        input_sentences,
        **generate_kwargs,
    )
    return outputs
