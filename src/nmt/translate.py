from __future__ import annotations

import argparse

from .inference import load_inference_bundle, translate_texts
from .runtime import get_tokenizers


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate English text to Chinese.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--text", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--beam-size", type=int, default=1, help="Beam size (1 = greedy).")
    parser.add_argument("--length-penalty", type=float, default=0.6)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
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
    src_tok, _, detok, _ = get_tokenizers(config)

    if args.text:
        output = translate_texts(
            model,
            src_vocab,
            tgt_vocab,
            [args.text],
            device=device,
            src_max_len=src_max_len,
            max_len=max_len,
            beam_size=args.beam_size,
            length_penalty=args.length_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            src_tokenizer=src_tok,
            detokenizer=detok,
        )[0]
        print(output.replace(" ", ""))
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
            beam_size=args.beam_size,
            length_penalty=args.length_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            src_tokenizer=src_tok,
            detokenizer=detok,
        )[0]
        print(f"ZH> {output.replace(' ', '')}")


if __name__ == "__main__":
    main()
