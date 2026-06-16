"""Download and prepare IWSLT English-to-Chinese data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

from .tokenizer import tokenize_en, tokenize_zh
from .vocab import Vocab


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare IWSLT en-zh data and vocabularies.")
    parser.add_argument("--dataset", default="IWSLT/iwslt2017", help="Hugging Face dataset name.")
    parser.add_argument("--config", default="iwslt2017-en-zh", help="Dataset config.")
    parser.add_argument("--output-dir", default="data/iwslt2017", help="Output directory.")
    parser.add_argument("--src-lang", default="en")
    parser.add_argument("--tgt-lang", default="zh")
    parser.add_argument("--src-vocab-size", type=int, default=32000)
    parser.add_argument("--tgt-vocab-size", type=int, default=12000)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Please install dependencies first: pip install -r requirements.txt") from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # IWSLT on Hugging Face is loaded through its dataset script, then normalized
    # into the simple {"src": English, "tgt": Chinese} JSONL format used below.
    dataset = load_dataset(args.dataset, args.config, trust_remote_code=args.trust_remote_code)
    split_limits = {
        "train": args.max_train_samples,
        "validation": args.max_valid_samples,
        "test": args.max_test_samples,
    }

    train_pairs = []
    for split_name in ("train", "validation", "test"):
        if split_name not in dataset:
            print(f"Skip missing split: {split_name}")
            continue
        pairs = list(iter_pairs(dataset[split_name], args.src_lang, args.tgt_lang, split_limits[split_name]))
        write_jsonl(output_dir / f"{split_name}.jsonl", pairs)
        print(f"Wrote {len(pairs):,} examples to {output_dir / f'{split_name}.jsonl'}")
        if split_name == "train":
            train_pairs = pairs

    if not train_pairs:
        raise SystemExit("No training examples were prepared; check dataset/config/language names.")

    # Build vocabularies only from the training split to avoid validation/test leakage.
    src_vocab = Vocab.build(
        (tokenize_en(src) for src, _ in train_pairs),
        max_size=args.src_vocab_size,
        min_freq=args.min_freq,
    )
    tgt_vocab = Vocab.build(
        (tokenize_zh(tgt) for _, tgt in train_pairs),
        max_size=args.tgt_vocab_size,
        min_freq=args.min_freq,
    )
    src_vocab.save(output_dir / "vocab.en.json")
    tgt_vocab.save(output_dir / "vocab.zh.json")
    print(f"Source vocab size: {len(src_vocab):,}")
    print(f"Target vocab size: {len(tgt_vocab):,}")


def iter_pairs(
    split,
    src_lang: str,
    tgt_lang: str,
    max_samples: int | None,
) -> Iterable[Tuple[str, str]]:
    count = 0
    for row in split:
        src, tgt = extract_pair(row, src_lang, tgt_lang)
        if src and tgt:
            yield src, tgt
            count += 1
            if max_samples is not None and count >= max_samples:
                break


def extract_pair(row: Dict[str, object], src_lang: str, tgt_lang: str) -> Tuple[str, str]:
    translation = row.get("translation", row)
    if isinstance(translation, str):
        translation = json.loads(translation)
    if not isinstance(translation, dict):
        raise ValueError(f"Unexpected translation row: {row}")
    return str(translation[src_lang]).strip(), str(translation[tgt_lang]).strip()


def write_jsonl(path: Path, pairs: Iterable[Tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for src, tgt in pairs:
            f.write(json.dumps({"src": src, "tgt": tgt}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
