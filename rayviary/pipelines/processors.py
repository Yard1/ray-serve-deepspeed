from typing import List, Union

import torch
from transformers import LogitsProcessor

MIN_BATCH_LEN = 4


class StopOnTokensLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        stopping_tokens: List[Union[List[int], int]],
        eos_token_id: Union[int, List[int]],
    ) -> None:
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        self.eos_token_id = eos_token_id
        self.stop_ids = [
            torch.LongTensor([stop_id] if not isinstance(stop_id, list) else stop_id)
            for stop_id in stopping_tokens
        ]
        self._stopped_batches = set()

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        for batch_index, batch in enumerate(input_ids):
            if len(batch) > MIN_BATCH_LEN and batch_index not in self._stopped_batches:
                for stop_id_index, _ in enumerate(self.stop_ids):
                    stop_id = self.stop_ids[stop_id_index].to(batch.device)
                    if len(batch) > len(stop_id) and batch[-len(stop_id) :].equal(
                        stop_id
                    ):
                        self._stopped_batches.add(batch_index)
            if batch_index in self._stopped_batches:
                num_tokens = scores.shape[1]
                scores[
                    batch_index,
                    [
                        token_id
                        for token_id in range(num_tokens)
                        if token_id not in self.eos_token_id
                    ],
                ] = -float("inf")
                for token_id in self.eos_token_id:
                    scores[batch_index, token_id] = 0
        return scores
