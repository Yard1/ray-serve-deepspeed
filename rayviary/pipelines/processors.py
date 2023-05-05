from typing import List, Union

import torch
from transformers import LogitsProcessor, StoppingCriteria

from ..logger import get_logger

MIN_ITERS = 4

logger = get_logger(__name__)


class StopOnEOS(StoppingCriteria):
    def __init__(self, eos_token_id: int) -> None:
        self.eos_token_id = eos_token_id

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        return all(batch[-1] == self.eos_token_id for batch in input_ids)


class StopOnTokensLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        stopping_tokens: List[Union[List[int], int]],
        eos_token_id: Union[int, List[int]],
        warmup_iters: int = MIN_ITERS,
    ) -> None:
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        self.eos_token_id = eos_token_id
        self.stop_ids = [
            torch.LongTensor([stop_id] if not isinstance(stop_id, list) else stop_id)
            for stop_id in stopping_tokens
        ]
        self._stopped_batches = set()
        self._iters = warmup_iters
        self._nulled_batch = None

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        # We want to let it run for at least a few iterations to
        # avoid stopping right away.
        self._iters -= 1
        if self._iters >= 0:
            return scores
        for batch_index, batch in enumerate(input_ids):
            if batch_index not in self._stopped_batches:
                for stop_id in self.stop_ids:
                    if len(batch) > len(stop_id) and batch[-len(stop_id) :].equal(
                        stop_id.to(batch.device)
                    ):
                        self._stopped_batches.add(batch_index)
            if batch_index in self._stopped_batches:
                if self._nulled_batch is None:
                    scores[batch_index, :] = -float("inf")
                    scores[batch_index, self.eos_token_id] = 0
                    self._nulled_batch = scores[batch_index]
                scores[batch_index] = self._nulled_batch
        return scores
