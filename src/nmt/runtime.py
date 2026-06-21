from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

from .model import TransformerNMT
from .tokenizer import detokenize_zh, join_spaced, tokenize_en, tokenize_zh, whitespace_tokenize
from .vocab import Vocab


def get_tokenizers(config: Dict[str, Any]):
    if bool(config.get("data", {}).get("pretokenized", False)):
        # space-join so BLEU can re-split into word tokens (the reference is space-separated)
        return whitespace_tokenize, whitespace_tokenize, join_spaced, whitespace_tokenize
    return tokenize_en, tokenize_zh, detokenize_zh, tokenize_zh


def get_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(config: Dict[str, Any], src_vocab: Vocab, tgt_vocab: Vocab) -> TransformerNMT:
    model_cfg = config.get("model", {})
    return TransformerNMT(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=int(model_cfg.get("d_model", 512)),
        n_heads=int(model_cfg.get("n_heads", 8)),
        num_encoder_layers=int(model_cfg.get("num_encoder_layers", 6)),
        num_decoder_layers=int(model_cfg.get("num_decoder_layers", 6)),
        dim_feedforward=int(model_cfg.get("dim_feedforward", 2048)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        max_len=int(model_cfg.get("max_len", 512)),
        pad_idx=tgt_vocab.pad_idx,
        norm_first=bool(model_cfg.get("norm_first", False)),
        tie_embeddings=bool(model_cfg.get("tie_embeddings", False)),
    )


def load_vocabs(config: Dict[str, Any]) -> tuple[Vocab, Vocab]:
    data_cfg = config["data"]
    src_vocab = Vocab.load(data_cfg["src_vocab_path"])
    tgt_vocab = Vocab.load(data_cfg["tgt_vocab_path"])
    return src_vocab, tgt_vocab


def load_checkpoint(path: str | Path, device: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(Path(path), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(Path(path), map_location=device)
