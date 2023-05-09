from typing import List, Union

import torch
from transformers import PreTrainedTokenizer

from ..models import Prompt


def get_special_token_id(tokenizer: PreTrainedTokenizer, key: str) -> int:
    token_ids = tokenizer.encode(key)
    return token_ids[0]


def truncate_to_first_stop_token(
    tokens: torch.LongTensor,
    stop_ids: List[Union[int, List[int]]],
) -> List[int]:
    """Truncate tokens up to the first stop_id."""
    if not stop_ids:
        return tokens
    stop_ids: List[torch.LongTensor] = [
        torch.LongTensor([stop_id] if not isinstance(stop_id, list) else stop_id)
        for stop_id in stop_ids
    ]
    for i in range(len(tokens)):
        for stop_id_index, _ in enumerate(stop_ids):
            stop_id = stop_ids[stop_id_index].to(tokens.device)
            if len(tokens) - i > len(stop_id) and tokens[i : len(stop_id) + i].equal(
                stop_id
            ):
                return tokens[:i]
    return tokens


def _construct_prompt(prompt: Union[str, Prompt], prompt_format: str) -> str:
    if isinstance(prompt, Prompt):
        if prompt.use_prompt_format and prompt_format:
            return prompt_format.format(instruction=prompt.prompt)
        else:
            return prompt.prompt
    return prompt_format.format(instruction=prompt) if prompt_format else prompt


def construct_prompts(
    prompts: Union[str, Prompt, List[str], List[Prompt]],
    prompt_format: str,
) -> List[str]:
    if not isinstance(prompts, list):
        prompts = [prompts]
    return [_construct_prompt(prompt, prompt_format) for prompt in prompts]


def tokenize_stopping_sequences_where_needed(
    tokenizer: PreTrainedTokenizer,
    stopping_sequences: List[Union[str, int, List[int]]],
) -> List[Union[List[int], int]]:
    """If any sequence is a string, tokenize it."""
    if not stopping_sequences:
        return None
    return [
        get_special_token_id(tokenizer, key) if isinstance(key, str) else key
        for key in stopping_sequences
    ]
