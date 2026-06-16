"""Vocabulary utilities for sequence-to-sequence translation."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Iterable, List, Sequence


PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"
SPECIAL_TOKENS = [PAD, UNK, BOS, EOS]


class Vocab:
    def __init__(self, tokens: Sequence[str]):
        unique_tokens: List[str] = []
        seen = set()
        for token in tokens:
            if token not in seen:
                unique_tokens.append(token)
                seen.add(token)
        for special in reversed(SPECIAL_TOKENS):
            if special not in seen:
                unique_tokens.insert(0, special)
        self.tokens = unique_tokens
        self.stoi = {token: idx for idx, token in enumerate(self.tokens)}

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def pad_idx(self) -> int:
        return self.stoi[PAD]

    @property
    def unk_idx(self) -> int:
        return self.stoi[UNK]

    @property
    def bos_idx(self) -> int:
        return self.stoi[BOS]

    @property
    def eos_idx(self) -> int:
        return self.stoi[EOS]

    def token_to_id(self, token: str) -> int:
        return self.stoi.get(token, self.unk_idx)

    def id_to_token(self, idx: int) -> str:
        if 0 <= idx < len(self.tokens):
            return self.tokens[idx]
        return UNK

    def encode(
        self,
        tokens: Sequence[str],
        *,
        add_bos: bool = True,
        add_eos: bool = True,
        max_len: int | None = None,
    ) -> List[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(self.bos_idx)
        ids.extend(self.token_to_id(token) for token in tokens)
        if add_eos:
            ids.append(self.eos_idx)
        if max_len is not None and len(ids) > max_len:
            ids = ids[:max_len]
            if add_eos:
                ids[-1] = self.eos_idx
        return ids

    def decode(
        self,
        ids: Sequence[int],
        *,
        skip_special: bool = True,
        stop_at_eos: bool = True,
    ) -> List[str]:
        tokens: List[str] = []
        for idx in ids:
            token = self.id_to_token(int(idx))
            if stop_at_eos and token == EOS:
                break
            if skip_special and token in SPECIAL_TOKENS:
                continue
            tokens.append(token)
        return tokens

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"tokens": self.tokens, "special_tokens": SPECIAL_TOKENS}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return cls(payload)
        return cls(payload["tokens"])

    @classmethod
    def build(
        cls,
        tokenized_sentences: Iterable[Sequence[str]],
        *,
        max_size: int | None = None,
        min_freq: int = 1,
    ) -> "Vocab":
        counter: Counter[str] = Counter()
        for tokens in tokenized_sentences:
            counter.update(tokens)
        sorted_tokens = [
            token
            for token, freq in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
            if freq >= min_freq and token not in SPECIAL_TOKENS
        ]
        if max_size is not None:
            sorted_tokens = sorted_tokens[: max(0, max_size - len(SPECIAL_TOKENS))]
        return cls([*SPECIAL_TOKENS, *sorted_tokens])
