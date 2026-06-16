"""Translate English sentences with a trained checkpoint."""

from __future__ import annotations

import argparse

from .inference import load_inference_bundle, translate_texts


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate English text to Chinese.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--text", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-len", type=int, default=None)
    args = parser.parse_args()

    model, config, src_vocab, tgt_vocab, device = load_inference_bundle(
        args.checkpoint,
        config_path=args.config,
        device_name=args.device,
    )
    data_cfg = config["data"]
    decode_cfg = config.get("decode", {})
    max_len = args.max_len or int(decode_cfg.get("max_len", 128))
    src_max_len = int(data_cfg.get("src_max_len", 128))

    if args.text:
        output = translate_texts(
            model,
            src_vocab,
            tgt_vocab,
            [args.text],
            device=device,
            src_max_len=src_max_len,
            max_len=max_len,
        )[0]
        print(output)
        return

    print("Interactive translation. Press Ctrl+C or enter an empty line to exit.")
    while True:
        try:
            text = input("EN> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not text:
            break
        output = translate_texts(
            model,
            src_vocab,
            tgt_vocab,
            [text],
            device=device,
            src_max_len=src_max_len,
            max_len=max_len,
        )[0]
        print(f"ZH> {output}")


if __name__ == "__main__":
    main()
