# Based on https://github.com/microsoft/DeepSpeedExamples/tree/master/inference/huggingface/text-generation

import argparse
import gc
from typing import List

import torch

from pipelines.dolly2_pipeline import DollyV2Pipeline
from pipelines.stablelm_pipeline import StableLMPipeline
from utils import timeit


@timeit
def init_model(
    args: argparse.Namespace, world_size: int, local_rank: int
) -> DollyV2Pipeline:
    """Initialize the deepspeed model"""
    # Lazy import so that the new cache location is used
    import deepspeed
    from deepspeed.runtime.utils import see_memory_usage
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

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
        tokenizer_kwargs = dict(padding_side="left")
    elif "stablelm" in args.name:
        tokenizer_kwargs = dict(padding_side="left")
    else:
        raise ValueError("Unsupported model")

    tokenizer = AutoTokenizer.from_pretrained(args.name, **tokenizer_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        args.name, low_cpu_mem_usage=True, torch_dtype=data_type, use_cache=True
    )

    if local_rank == 0:
        see_memory_usage("after init", True)

    ds_kwargs = dict()

    gc.collect()

    from transformers import GPTNeoXForCausalLM, GPTNeoXLayer

    if args.ds_inference:
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
            save_mp_checkpoint_path=args.save_mp_checkpoint_path,
            **ds_kwargs,
        )

    if local_rank == 0:
        see_memory_usage("after init_inference", True)

    device = torch.device(f"cuda:{local_rank}")
    model.cuda().to(device)
    # Add this attribute for compatibility with the pipeline
    model.device = device

    if "dolly-v2" in args.name:
        pipe = DollyV2Pipeline(model=model, tokenizer=tokenizer, device=device)
    elif "stablelm" in args.name:
        pipe = StableLMPipeline(model=model, tokenizer=tokenizer, device=device)

    print(pipe)

    # Warmup
    print(generate(["Write a short story."], pipe, max_new_tokens=256))
    print(
        generate(
            ["Write a short story.", "Write a short story."], pipe, max_new_tokens=256
        )
    )

    return pipe


@timeit
@torch.inference_mode()
def generate(
    input_sentences: List[str], pipe: DollyV2Pipeline, **generate_kwargs
) -> List[str]:
    """Generate predictions using a Pipeline"""
    outputs = pipe(
        input_sentences,
        **generate_kwargs,
    )
    return outputs
