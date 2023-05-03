from typing import List, Union

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
    tokens: List[int],
    stop_ids: List[Union[int, List[int]]],
    eos_token_id: Union[int, List[int]],
) -> List[int]:
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    for i, _ in enumerate(stop_ids):
        last_index = -1
        while len(tokens) + last_index + 1 > 0 and any(
            token == tokens[last_index] for token in eos_token_id
        ):
            last_index -= 1
        tokens = tokens[:last_index]
        stop_ids[i] = stop_ids[i].to(tokens.device)
        last_tokens = tokens[-len(stop_ids[i]) :]
        if last_tokens.equal(stop_ids[i]):
            tokens = tokens[: -len(stop_ids[i])]
            break
    return tokens
