from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch

from .config import load_config
from .model import beam_search_decode, greedy_decode
from .runtime import build_model, get_device, load_checkpoint, load_vocabs
from .tokenizer import detokenize_zh, tokenize_en


def load_inference_bundle(
    checkpoint_path: str | Path,
    *,
    config_path: str | Path | None = None,
    device_name: str | None = None,
) -> tuple[torch.nn.Module, Dict[str, Any], Any, Any, torch.device]:
    device = get_device(device_name)
    checkpoint = load_checkpoint(checkpoint_path, device)
    config = load_config(config_path) if config_path else checkpoint["config"]
    src_vocab, tgt_vocab = load_vocabs(config)
    model = build_model(config, src_vocab, tgt_vocab).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, config, src_vocab, tgt_vocab, device


@torch.no_grad()
def translate_texts(
    model: torch.nn.Module,
    src_vocab,
    tgt_vocab,
    texts: List[str],
    *,
    device: torch.device,
    src_max_len: int,
    max_len: int,
    beam_size: int = 1,
    length_penalty: float = 0.6,
    no_repeat_ngram_size: int = 0,
    src_tokenizer=tokenize_en,
    detokenizer=detokenize_zh,
) -> List[str]:
    encoded = [
        torch.tensor(
            src_vocab.encode(src_tokenizer(text), add_bos=True, add_eos=True, max_len=src_max_len),
            dtype=torch.long,
        )
        for text in texts
    ]
    src = torch.nn.utils.rnn.pad_sequence(encoded, batch_first=True, padding_value=src_vocab.pad_idx).to(device)
    if beam_size and beam_size > 1:
        generated = beam_search_decode(
            model, src, bos_idx=tgt_vocab.bos_idx, eos_idx=tgt_vocab.eos_idx, max_len=max_len,
            beam_size=beam_size, length_penalty=length_penalty, no_repeat_ngram_size=no_repeat_ngram_size,
        )
    else:
        generated = greedy_decode(model, src, bos_idx=tgt_vocab.bos_idx, eos_idx=tgt_vocab.eos_idx, max_len=max_len)
    outputs = []
    for ids in generated.tolist():
        # Drop <bos>/<eos>/<pad> and join the target tokens for display.
        tokens = tgt_vocab.decode(ids, skip_special=True, stop_at_eos=True)
        outputs.append(detokenizer(tokens))
    return outputs
