from typing import List, Union

import torch
from transformers import PreTrainedTokenizer


def get_special_token_id(tokenizer: PreTrainedTokenizer, key: str) -> int:
    """Gets the token ID for a given string that has been added to the tokenizer as a special token.
    When training, we configure the tokenizer so that the sequences like "### Instruction:" and "### End" are
    treated specially and converted to a single, new token.  This retrieves the token ID each of these keys map to.
    Args:
        tokenizer (PreTrainedTokenizer): the tokenizer
        key (str): the key to convert to a single token
    Raises:
        RuntimeError: if more than one ID was generated
    Returns:
        int: the token ID for the given key
    """
    token_ids = tokenizer.encode(key)
    if len(token_ids) > 2:
        raise ValueError(
            f"Expected only a single token for '{key}' but found {token_ids}"
        )
    return token_ids[0]


def remove_dangling_stop_tokens(
    tokens: torch.LongTensor,
    stop_ids: List[Union[int, List[int]]],
) -> List[int]:
    if not stop_ids:
        return tokens
    stop_ids: List[torch.LongTensor] = [
        torch.LongTensor([stop_id] if not isinstance(stop_id, list) else stop_id)
        for stop_id in stop_ids
    ]
    last_token_is_stop_token = True
    while last_token_is_stop_token:
        for stop_id_index, _ in enumerate(stop_ids):
            stop_id = stop_ids[stop_id_index].to(tokens.device)
            if len(tokens) > len(stop_id) and tokens[-len(stop_id) :].equal(stop_id):
                tokens = tokens[: -len(stop_id)]
                last_token_is_stop_token = True
                break
            else:
                last_token_is_stop_token = False
    return tokens
