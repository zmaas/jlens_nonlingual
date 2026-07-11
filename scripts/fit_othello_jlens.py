#!/usr/bin/env python3
"""Fit Anthropic's Jacobian Lens on synthetic legal Othello games."""

from __future__ import annotations

import argparse
from pathlib import Path

from othello_common import generate_games, load_model, parse_layers, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", help="Local synthetic_model.pth (avoids Hub download)")
    parser.add_argument("--n-prompts", type=int, default=10)
    parser.add_argument("--max-seq-len", type=int, default=59)
    parser.add_argument("--source-layers", default="0,1,2,3,4,5,6")
    parser.add_argument("--target-layer", type=int, default=7)
    parser.add_argument("--dim-batch", type=int, default=128)
    parser.add_argument("--skip-first", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="out/coffee/othello_jlens.pt")
    parser.add_argument("--checkpoint-path", default="out/coffee/othello_fit_ckpt.pt")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    import jlens
    from jlens.adapters import TransformerLensLensModel

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model = TransformerLensLensModel(load_model(args.device, args.checkpoint))
    prompts = generate_games(args.n_prompts, seed=args.seed, min_length=args.skip_first + 2)
    lens = jlens.fit(
        model,
        prompts,
        source_layers=parse_layers(args.source_layers),
        target_layer=args.target_layer,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        skip_first=args.skip_first,
        checkpoint_path=args.checkpoint_path,
        resume=not args.no_resume,
    )
    lens.save(str(out))
    write_json(
        out.with_suffix(".metadata.json"),
        {
            "model": "OthelloGPT synthetic TransformerLens checkpoint",
            "source_layers": lens.source_layers,
            "target_layer": args.target_layer,
            "n_prompts": lens.n_prompts,
            "max_seq_len": args.max_seq_len,
            "dim_batch": args.dim_batch,
            "skip_first": args.skip_first,
            "seed": args.seed,
        },
    )
    print(f"saved {out}")


if __name__ == "__main__":
    main()
