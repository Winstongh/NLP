"""Convert the provided cmn-eng-simple dataset (already tokenized: English BPE,
Chinese jieba word-segmented, tab-separated) into the project's jsonl format,
and build vocabularies from the whitespace tokens.

Input : data/cmn-eng-simple/{training,validation,testing}.txt
Output: data/cmn/{train,validation,test}.jsonl + vocab.en.json + vocab.zh.json
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import json

from nmt.vocab import Vocab

SRC = ROOT / "data" / "cmn-eng-simple"
OUT = ROOT / "data" / "cmn"


def convert(in_name: str, out_name: str):
    pairs = []
    for line in (SRC / in_name).open(encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        en, cn = parts[0].strip(), parts[1].strip()
        if en and cn:
            pairs.append((en, cn))
    with (OUT / out_name).open("w", encoding="utf-8") as f:
        for en, cn in pairs:
            f.write(json.dumps({"src": en, "tgt": cn}, ensure_ascii=False) + "\n")
    return pairs


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train = convert("training.txt", "train.jsonl")
    valid = convert("validation.txt", "validation.jsonl")
    test = convert("testing.txt", "test.jsonl")
    print(f"train={len(train)}  valid={len(valid)}  test={len(test)}")

    # data is already tokenized -> split on whitespace
    sv = Vocab.build((en.split() for en, _ in train), min_freq=1)
    tv = Vocab.build((cn.split() for _, cn in train), min_freq=1)
    sv.save(OUT / "vocab.en.json")
    tv.save(OUT / "vocab.zh.json")
    print(f"en vocab={len(sv)}  zh vocab={len(tv)}  -> {OUT}")

    # quick sanity: avg lengths
    import statistics
    el = statistics.mean(len(en.split()) for en, _ in train)
    cl = statistics.mean(len(cn.split()) for _, cn in train)
    print(f"avg en tokens={el:.1f}  avg zh words={cl:.1f}")


if __name__ == "__main__":
    main()
