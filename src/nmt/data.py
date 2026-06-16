"""Dataset and dataloader helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List

import torch
from torch.utils.data import DataLoader, Dataset

from .tokenizer import tokenize_en, tokenize_zh
from .vocab import Vocab


TokenizeFn = Callable[[str], List[str]]


class TranslationDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        src_vocab: Vocab,
        tgt_vocab: Vocab,
        *,
        src_tokenizer: TokenizeFn = tokenize_en,
        tgt_tokenizer: TokenizeFn = tokenize_zh,
        src_max_len: int = 128,
        tgt_max_len: int = 128,
        max_samples: int | None = None,
    ) -> None:
        self.path = Path(path)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_max_len = src_max_len
        self.tgt_max_len = tgt_max_len
        self.examples = list(read_jsonl(self.path, max_samples=max_samples))
        if not self.examples:
            raise ValueError(f"No examples found in {self.path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        item = self.examples[index]
        src_text = item["src"]
        tgt_text = item["tgt"]
        src_ids = self.src_vocab.encode(
            self.src_tokenizer(src_text),
            add_bos=True,
            add_eos=True,
            max_len=self.src_max_len,
        )
        tgt_ids = self.tgt_vocab.encode(
            self.tgt_tokenizer(tgt_text),
            add_bos=True,
            add_eos=True,
            max_len=self.tgt_max_len,
        )
        return {
            "src": torch.tensor(src_ids, dtype=torch.long),
            "tgt": torch.tensor(tgt_ids, dtype=torch.long),
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def read_jsonl(path: str | Path, *, max_samples: int | None = None) -> Iterable[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if max_samples is not None and line_number > max_samples:
                break
            if not line.strip():
                continue
            item = json.loads(line)
            if "src" not in item or "tgt" not in item:
                raise ValueError(f"{path}:{line_number} must contain 'src' and 'tgt'")
            yield {"src": str(item["src"]), "tgt": str(item["tgt"])}


def collate_translation_batch(batch: List[Dict[str, torch.Tensor | str]], pad_idx: int) -> Dict[str, object]:
    # Pad variable-length sentences in a batch, then split target for teacher forcing.
    src_tensors = [item["src"] for item in batch]
    tgt_tensors = [item["tgt"] for item in batch]
    src = torch.nn.utils.rnn.pad_sequence(src_tensors, batch_first=True, padding_value=pad_idx)
    tgt = torch.nn.utils.rnn.pad_sequence(tgt_tensors, batch_first=True, padding_value=pad_idx)
    return {
        "src": src,
        "tgt_input": tgt[:, :-1],
        "tgt_output": tgt[:, 1:],
        "src_text": [item["src_text"] for item in batch],
        "tgt_text": [item["tgt_text"] for item in batch],
    }


def make_dataloader(
    dataset: TranslationDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=lambda batch: collate_translation_batch(batch, dataset.tgt_vocab.pad_idx),
    )
