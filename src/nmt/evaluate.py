from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch import nn

from .data import TranslationDataset, make_dataloader, read_jsonl
from .inference import load_inference_bundle, translate_texts
from .metrics import corpus_bleu
from .runtime import get_tokenizers
from .tokenizer import detokenize_zh, tokenize_en, tokenize_zh
from .train import evaluate_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Transformer NMT checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None, help="Override config path; otherwise uses config stored in checkpoint.")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--bleu-samples", type=int, default=2000)
    parser.add_argument("--beam-size", type=int, default=1, help="Beam size for decoding (1 = greedy).")
    parser.add_argument("--length-penalty", type=float, default=0.6, help="Beam search length penalty (GNMT).")
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0, help="Block repeated n-grams (0 = off).")
    parser.add_argument("--output", default=None, help="Optional JSON metrics output path.")
    args = parser.parse_args()

    metrics = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        split=args.split,
        data_path=args.data_path,
        batch_size=args.batch_size,
        device_name=args.device,
        max_samples=args.max_samples,
        bleu_samples=args.bleu_samples,
        beam_size=args.beam_size,
        length_penalty=args.length_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate_checkpoint(
    *,
    checkpoint_path: str | Path,
    config_path: str | Path | None,
    split: str,
    data_path: str | Path | None,
    batch_size: int | None,
    device_name: str | None,
    max_samples: int | None,
    bleu_samples: int,
    beam_size: int = 1,
    length_penalty: float = 0.6,
    no_repeat_ngram_size: int = 0,
) -> Dict[str, Any]:
    model, config, src_vocab, tgt_vocab, device = load_inference_bundle(
        checkpoint_path,
        config_path=config_path,
        device_name=device_name,
    )
    src_tok, tgt_tok, detok, bleu_tok = get_tokenizers(config)
    data_cfg = config["data"]
    train_cfg = config.get("train", {})
    decode_cfg = config.get("decode", {})
    eval_path = Path(data_path or data_cfg[f"{split}_path" if split == "test" else "valid_path"])

    dataset = TranslationDataset(
        eval_path,
        src_vocab,
        tgt_vocab,
        src_tokenizer=src_tok,
        tgt_tokenizer=tgt_tok,
        src_max_len=int(data_cfg.get("src_max_len", 128)),
        tgt_max_len=int(data_cfg.get("tgt_max_len", 128)),
        max_samples=max_samples,
    )
    loader = make_dataloader(
        dataset,
        batch_size=batch_size or int(train_cfg.get("batch_size", 64)),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
    )
    criterion = nn.CrossEntropyLoss(
        ignore_index=tgt_vocab.pad_idx,
        label_smoothing=float(train_cfg.get("label_smoothing", 0.0)),
    )
    loss_stats = evaluate_loss(model, loader, criterion, device, pad_idx=tgt_vocab.pad_idx)
    generation = evaluate_generation(
        model,
        src_vocab,
        tgt_vocab,
        eval_path,
        device=device,
        src_max_len=int(data_cfg.get("src_max_len", 128)),
        max_len=int(decode_cfg.get("max_len", 128)),
        max_samples=min(bleu_samples, len(dataset)),
        batch_size=batch_size or int(train_cfg.get("batch_size", 64)),
        beam_size=beam_size,
        length_penalty=length_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        src_tokenizer=src_tok,
        detokenizer=detok,
        bleu_tokenizer=bleu_tok,
    )
    return {
        "split": split,
        "num_examples": len(dataset),
        "beam_size": beam_size,
        "loss": round(loss_stats["loss"], 6),
        "token_accuracy": round(loss_stats["accuracy"], 6),
        "bleu": round(generation["bleu"], 4),
        "samples": generation["samples"],
    }


def evaluate_generation(
    model,
    src_vocab,
    tgt_vocab,
    path: Path,
    *,
    device: torch.device,
    src_max_len: int,
    max_len: int,
    max_samples: int,
    batch_size: int,
    beam_size: int = 1,
    length_penalty: float = 0.6,
    no_repeat_ngram_size: int = 0,
    src_tokenizer=tokenize_en,
    detokenizer=detokenize_zh,
    bleu_tokenizer=tokenize_zh,
) -> Dict[str, Any]:
    examples = list(read_jsonl(path, max_samples=max_samples))
    hypotheses: List[List[str]] = []
    references: List[List[str]] = []
    sample_rows: List[Dict[str, str]] = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        outputs = translate_texts(
            model,
            src_vocab,
            tgt_vocab,
            [item["src"] for item in batch],
            device=device,
            src_max_len=src_max_len,
            max_len=max_len,
            beam_size=beam_size,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            src_tokenizer=src_tokenizer,
            detokenizer=detokenizer,
        )
        for item, output in zip(batch, outputs):
            hypotheses.append(bleu_tokenizer(output))
            references.append(bleu_tokenizer(item["tgt"]))
            if len(sample_rows) < 5:
                sample_rows.append({"src": item["src"], "ref": item["tgt"], "hyp": output})
    return {"bleu": corpus_bleu(hypotheses, references), "samples": sample_rows}


if __name__ == "__main__":
    main()
