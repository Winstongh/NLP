"""Training entrypoint for the Transformer NMT model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time
from typing import Any, Dict

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .config import load_config, save_config
from .data import TranslationDataset, make_dataloader
from .metrics import token_accuracy
from .runtime import build_model, get_device, load_checkpoint, load_vocabs


class NoamScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, d_model: int, warmup_steps: int, factor: float = 1.0):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.step_num = 0

    def step(self) -> float:
        self.step_num += 1
        lr = self.rate()
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def rate(self) -> float:
        step = max(self.step_num, 1)
        # Transformer schedule from "Attention Is All You Need":
        # linear warmup followed by inverse square-root decay.
        return self.factor * (self.d_model ** -0.5) * min(step ** -0.5, step * (self.warmup_steps ** -1.5))

    def state_dict(self) -> Dict[str, Any]:
        return {"step_num": self.step_num}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.step_num = int(state.get("step_num", 0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Transformer NMT model.")
    parser.add_argument("--config", default="configs/rtx4090.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from.")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, device_name=args.device, resume=args.resume, max_train_samples=args.max_train_samples, max_valid_samples=args.max_valid_samples)


def train(
    config: Dict[str, Any],
    *,
    device_name: str | None = None,
    resume: str | None = None,
    max_train_samples: int | None = None,
    max_valid_samples: int | None = None,
) -> None:
    set_seed(int(config.get("seed", 42)))
    device = get_device(device_name)
    src_vocab, tgt_vocab = load_vocabs(config)
    data_cfg = config["data"]
    train_cfg = config["train"]
    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.yaml")

    train_dataset = TranslationDataset(
        data_cfg["train_path"],
        src_vocab,
        tgt_vocab,
        src_max_len=int(data_cfg.get("src_max_len", 128)),
        tgt_max_len=int(data_cfg.get("tgt_max_len", 128)),
        max_samples=max_train_samples,
    )
    valid_dataset = TranslationDataset(
        data_cfg["valid_path"],
        src_vocab,
        tgt_vocab,
        src_max_len=int(data_cfg.get("src_max_len", 128)),
        tgt_max_len=int(data_cfg.get("tgt_max_len", 128)),
        max_samples=max_valid_samples,
    )
    train_loader = make_dataloader(
        train_dataset,
        batch_size=int(train_cfg.get("batch_size", 64)),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 0)),
    )
    valid_loader = make_dataloader(
        valid_dataset,
        batch_size=int(train_cfg.get("batch_size", 64)),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
    )

    model = build_model(config, src_vocab, tgt_vocab).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1.0)),
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )
    scheduler = NoamScheduler(
        optimizer,
        d_model=int(config["model"].get("d_model", 512)),
        warmup_steps=int(train_cfg.get("warmup_steps", 4000)),
        factor=float(train_cfg.get("learning_rate", 1.0)),
    )
    criterion = nn.CrossEntropyLoss(
        ignore_index=tgt_vocab.pad_idx,
        label_smoothing=float(train_cfg.get("label_smoothing", 0.1)),
    )
    use_amp = bool(train_cfg.get("amp", False)) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    start_epoch = 1
    best_valid_loss = float("inf")
    if resume:
        checkpoint = load_checkpoint(resume, device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint.get("scheduler_state", {}))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_valid_loss = float(checkpoint.get("best_valid_loss", best_valid_loss))

    print(f"Device: {device}")
    print(f"Train examples: {len(train_dataset):,}; valid examples: {len(valid_dataset):,}")
    print(f"Source vocab: {len(src_vocab):,}; target vocab: {len(tgt_vocab):,}")

    log_path = output_dir / "train_log.jsonl"
    for epoch in range(start_epoch, int(train_cfg.get("epochs", 30)) + 1):
        epoch_start = time.time()
        train_stats = run_train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            scaler,
            device,
            pad_idx=tgt_vocab.pad_idx,
            grad_clip=float(train_cfg.get("grad_clip", 1.0)),
            use_amp=use_amp,
            log_every=int(train_cfg.get("log_every", 100)),
        )
        valid_stats = evaluate_loss(model, valid_loader, criterion, device, pad_idx=tgt_vocab.pad_idx)
        elapsed = time.time() - epoch_start
        row = {"epoch": epoch, "elapsed_sec": round(elapsed, 2), **prefix_keys("train", train_stats), **prefix_keys("valid", valid_stats)}
        append_jsonl(log_path, row)
        print(
            f"Epoch {epoch}: train_loss={train_stats['loss']:.4f}, "
            f"valid_loss={valid_stats['loss']:.4f}, valid_acc={valid_stats['accuracy']:.4f}, "
            f"time={elapsed:.1f}s"
        )
        is_best = valid_stats["loss"] < best_valid_loss
        if is_best:
            best_valid_loss = valid_stats["loss"]
        if epoch % int(train_cfg.get("save_every_epochs", 1)) == 0 or is_best:
            save_checkpoint(
                output_dir / "last.pt",
                model,
                optimizer,
                scheduler,
                config,
                epoch,
                best_valid_loss,
            )
        if is_best:
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                config,
                epoch,
                best_valid_loss,
            )


def run_train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scheduler: NoamScheduler,
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    *,
    pad_idx: int,
    grad_clip: float,
    use_amp: bool,
    log_every: int,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0.0
    progress = tqdm(loader, desc="train", leave=False)
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(progress, start=1):
        src = batch["src"].to(device)
        tgt_input = batch["tgt_input"].to(device)
        tgt_output = batch["tgt_output"].to(device)
        token_count = int(tgt_output.ne(pad_idx).sum().item())
        with autocast(enabled=use_amp):
            # Teacher forcing: decoder input is gold target shifted right;
            # tgt_output is the next-token supervision signal.
            logits = model(src, tgt_input)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        lr = scheduler.step()

        total_loss += float(loss.item()) * token_count
        total_tokens += token_count
        total_correct += token_accuracy(logits.detach(), tgt_output, pad_idx) * token_count
        if step % log_every == 0:
            progress.set_postfix(loss=f"{total_loss / max(total_tokens, 1):.4f}", lr=f"{lr:.2e}")
    return {"loss": total_loss / max(total_tokens, 1), "accuracy": total_correct / max(total_tokens, 1)}


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader, criterion: nn.Module, device: torch.device, *, pad_idx: int) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0.0
    for batch in tqdm(loader, desc="eval", leave=False):
        src = batch["src"].to(device)
        tgt_input = batch["tgt_input"].to(device)
        tgt_output = batch["tgt_output"].to(device)
        logits = model(src, tgt_input)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
        token_count = int(tgt_output.ne(pad_idx).sum().item())
        total_loss += float(loss.item()) * token_count
        total_tokens += token_count
        total_correct += token_accuracy(logits, tgt_output, pad_idx) * token_count
    return {"loss": total_loss / max(total_tokens, 1), "accuracy": total_correct / max(total_tokens, 1)}


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: NoamScheduler,
    config: Dict[str, Any],
    epoch: int,
    best_valid_loss: float,
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "config": config,
        "epoch": epoch,
        "best_valid_loss": best_valid_loss,
    }
    torch.save(payload, path)
    print(f"Saved checkpoint: {path}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prefix_keys(prefix: str, values: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


if __name__ == "__main__":
    main()
