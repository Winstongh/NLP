from __future__ import annotations

from collections import Counter
import math
from typing import Iterable, List, Sequence

import torch


def token_accuracy(logits: torch.Tensor, targets: torch.Tensor, pad_idx: int) -> float:
    predictions = logits.argmax(dim=-1)
    mask = targets.ne(pad_idx)
    total = int(mask.sum().item())
    if total == 0:
        return 0.0
    correct = int(predictions.eq(targets).masked_select(mask).sum().item())
    return correct / total


def corpus_bleu(
    hypotheses: Sequence[Sequence[str]],
    references: Sequence[Sequence[str]],
    *,
    max_n: int = 4,
) -> float:
    """Compute a small smoothed corpus BLEU score in the 0-100 range."""

    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must have the same length")
    if not hypotheses:
        return 0.0

    clipped = [0] * max_n
    total = [0] * max_n
    hyp_len = 0
    ref_len = 0
    for hyp, ref in zip(hypotheses, references):
        hyp = list(hyp)
        ref = list(ref)
        hyp_len += len(hyp)
        ref_len += len(ref)
        for n in range(1, max_n + 1):
            hyp_counts = _ngram_counts(hyp, n)
            ref_counts = _ngram_counts(ref, n)
            clipped[n - 1] += sum(min(count, ref_counts[ngram]) for ngram, count in hyp_counts.items())
            total[n - 1] += max(len(hyp) - n + 1, 0)

    if hyp_len == 0:
        return 0.0
    precisions = [(clipped[i] + 1.0) / (total[i] + 1.0) for i in range(max_n)]
    brevity_penalty = 1.0 if hyp_len > ref_len else math.exp(1.0 - ref_len / max(hyp_len, 1))
    bleu = brevity_penalty * math.exp(sum(math.log(p) for p in precisions) / max_n)
    return bleu * 100.0


def _ngram_counts(tokens: List[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(max(len(tokens) - n + 1, 0)))
