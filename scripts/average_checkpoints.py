"""Average the weights of several checkpoints (a standard Transformer-MT trick
that reliably gains a bit of BLEU for free).

Requires per-epoch checkpoints, e.g. trained with `train.save_epoch_checkpoints: true`.

Example — average the last 5 epochs:
    python scripts/average_checkpoints.py \
        --checkpoints checkpoints/cmn/epoch_{51,52,53,54,55,56,57,58,59,60}.pt \
        --output checkpoints/cmn/avg.pt
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Average model weights across checkpoints.")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Checkpoint .pt files to average.")
    parser.add_argument("--output", required=True, help="Output averaged checkpoint path.")
    args = parser.parse_args()

    states = []
    base = None
    for path in args.checkpoints:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        states.append(ckpt["model_state"])
        if base is None:
            base = ckpt

    averaged = {}
    for key, ref in states[0].items():
        if ref.is_floating_point():
            acc = sum(state[key].float() for state in states) / len(states)
            averaged[key] = acc.to(ref.dtype)
        else:
            averaged[key] = ref.clone()  # int buffers: keep first

    base["model_state"] = averaged
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, args.output)
    print(f"Averaged {len(states)} checkpoints -> {args.output}")


if __name__ == "__main__":
    main()
