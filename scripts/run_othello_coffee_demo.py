#!/usr/bin/env python3
"""One-command Phase 1 OthelloGPT coffee demo."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", help="Local synthetic_model.pth (avoids Hub download)")
    parser.add_argument("--n-prompts", type=int, default=10)
    parser.add_argument("--n-games", type=int, default=25)
    parser.add_argument("--source-layers", default="0,1,2,3,4,5,6")
    parser.add_argument("--target-layer", type=int, default=7)
    parser.add_argument("--dim-batch", type=int, default=128)
    parser.add_argument("--out-dir", default="out/coffee")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    here, out = Path(__file__).resolve().parent, Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    common = ["--device", args.device]
    if args.checkpoint:
        common += ["--checkpoint", args.checkpoint]
    fit = [
        sys.executable,
        str(here / "fit_othello_jlens.py"),
        *common,
        "--n-prompts",
        str(args.n_prompts),
        "--source-layers",
        args.source_layers,
        "--target-layer",
        str(args.target_layer),
        "--dim-batch",
        str(args.dim_batch),
        "--out",
        str(out / "othello_jlens.pt"),
        "--checkpoint-path",
        str(out / "othello_fit_ckpt.pt"),
    ]
    if args.no_resume:
        fit.append("--no-resume")
    subprocess.run(fit, check=True)
    subprocess.run(
        [
            sys.executable,
            str(here / "eval_othello_jlens.py"),
            *common,
            "--lens",
            str(out / "othello_jlens.pt"),
            "--n-games",
            str(args.n_games),
            "--out",
            str(out / "othello_eval.json"),
            "--markdown",
            str(out / "othello_summary.md"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
