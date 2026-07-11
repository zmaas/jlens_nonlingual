#!/usr/bin/env python3
"""Run the three pre-coffee Othello workspace experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint")
    parser.add_argument("--lens", default="out/coffee_v2/othello_jlens.pt")
    parser.add_argument("--out-dir", default="out/precoffee")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    out = Path(args.out_dir)
    common = ["--device", args.device, "--lens", args.lens]
    if args.checkpoint:
        common += ["--checkpoint", args.checkpoint]
    sizes = {
        "legal_games": 10 if args.quick else 100,
        "steering_boards": 4 if args.quick else 40,
        "probe_games": 10 if args.quick else 60,
        "ablation_games": 3 if args.quick else 20,
        "bootstrap": 100 if args.quick else 1000,
    }
    commands = [
        [
            sys.executable,
            str(here / "eval_othello_legality.py"),
            *common,
            "--n-games",
            str(sizes["legal_games"]),
            "--n-bootstrap",
            str(sizes["bootstrap"]),
            "--out-dir",
            str(out / "legal"),
        ],
        [
            sys.executable,
            str(here / "intervene_othello_jspace.py"),
            *common,
            "--n-boards",
            str(sizes["steering_boards"]),
            "--n-bootstrap",
            str(sizes["bootstrap"]),
            "--out-dir",
            str(out / "steering"),
        ],
        [
            sys.executable,
            str(here / "analyze_othello_workspace_split.py"),
            *common,
            "--n-probe-games",
            str(sizes["probe_games"]),
            "--n-ablation-games",
            str(sizes["ablation_games"]),
            "--n-bootstrap",
            str(sizes["bootstrap"]),
            "--out-dir",
            str(out / "workspace"),
        ],
    ]
    for command in commands:
        print("running:", " ".join(command), flush=True)
        subprocess.run(command, check=True)
    subprocess.run(
        [
            sys.executable,
            str(here / "summarize_othello_precoffee.py"),
            "--out-dir",
            str(out),
        ],
        check=True,
    )
    print(f"all pre-coffee artifacts saved under {out}")


if __name__ == "__main__":
    main()
