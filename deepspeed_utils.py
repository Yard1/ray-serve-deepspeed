# Based on https://github.com/microsoft/DeepSpeedExamples/tree/master/inference/huggingface/text-generation

import argparse
import gc
from typing import TYPE_CHECKING, List

from utils import timeit

if TYPE_CHECKING:
    from transformers import Pipeline


@timeit
def init_model(
    args: argparse.Namespace, world_size: int, local_rank: int
) -> "Pipeline":
    """Initialize the deepspeed model"""
    # Lazy import so that the new cache location is used
    import deepspeed
    import torch
    from deepspeed.runtime.utils import see_memory_usage
    from transformers import pipeline

    # from deepspeed_pipeline import DSPipeline

    data_type = getattr(torch, args.dtype)

    if local_rank == 0:
        see_memory_usage("before init", True)

    pipe = pipeline(
        model=args.name,
        torch_dtype=data_type,
        trust_remote_code=True,
        model_kwargs=dict(low_cpu_mem_usage=True),
    )
    # pipe = DSPipeline(
    #     model_name=args.name,
    #     dtype=data_type,
    #     is_meta=False,
    #     device=local_rank,
    #     #repo_root=args.repo_root,
    # )
    if local_rank == 0:
        see_memory_usage("after init", True)

    ds_kwargs = dict()

    gc.collect()

    from transformers import GPTNeoXLayer

    if args.ds_inference:
        pipe.model = deepspeed.init_inference(
            pipe.model,
            dtype=data_type,
            mp_size=world_size,
            replace_with_kernel_inject=args.use_kernel,
            injection_policy={GPTNeoXLayer: ("attention.dense", "mlp.dense_4h_to_h")},
            max_tokens=args.max_tokens,
            save_mp_checkpoint_path=args.save_mp_checkpoint_path,
            **ds_kwargs,
        )

    if local_rank == 0:
        see_memory_usage("after init_inference", True)

    pipe.device = torch.device(f"cuda:{local_rank}")
    pipe.model.cuda().to(pipe.device)
    # Add this attribute for compatibility with the pipeline
    pipe.model.device = pipe.device

    return pipe


@timeit
def generate(
    input_sentences: List[str], pipe: "Pipeline", **generate_kwargs
) -> List[str]:
    """Generate predictions using a Pipeline"""
    outputs = pipe(
        input_sentences,
        # batch_size=len(input_sentences),
        **generate_kwargs,
    )
    return outputs
