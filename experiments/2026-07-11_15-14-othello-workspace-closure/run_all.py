#!/usr/bin/env python3
"""Run Othello J-lens v2, the three pre-coffee tests, and closure tests."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
LOG_DIR = EXPERIMENT_DIR / "logs" / "end_to_end"
MANIFEST_PATH = LOG_DIR / "manifest.json"


def _write_manifest(manifest: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    temporary = MANIFEST_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(MANIFEST_PATH)


def _run_stage(name: str, command: list[str], manifest: dict, *, dry_run: bool) -> None:
    printable = " ".join(command)
    print(f"\n=== {name} ===\n{printable}", flush=True)
    if dry_run:
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{name}.log"
    stage = {
        "name": name,
        "command": command,
        "log": str(log_path.relative_to(REPO_ROOT)),
        "started_at_unix": time.time(),
        "status": "running",
    }
    manifest["stages"].append(stage)
    _write_manifest(manifest)

    with log_path.open("w") as log_file:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()

    stage["ended_at_unix"] = time.time()
    stage["elapsed_seconds"] = stage["ended_at_unix"] - stage["started_at_unix"]
    stage["return_code"] = return_code
    stage["status"] = "complete" if return_code == 0 else "failed"
    _write_manifest(manifest)
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def _commands(args) -> list[tuple[str, list[str]]]:
    python = args.python
    checkpoint = ["--checkpoint", args.checkpoint] if args.checkpoint else []
    coffee_out = Path(args.coffee_out)
    lens_path = coffee_out / "othello_jlens.pt"

    if args.quick:
        n_prompts, n_games, dim_batch = 3, 10, 32
    else:
        n_prompts, n_games, dim_batch = args.n_prompts, args.n_games, args.dim_batch

    commands: list[tuple[str, list[str]]] = []
    if not args.skip_v2:
        coffee = [
            python,
            str(SCRIPTS_DIR / "run_othello_coffee_demo.py"),
            "--device",
            args.device,
            *checkpoint,
            "--n-prompts",
            str(n_prompts),
            "--n-games",
            str(n_games),
            "--source-layers",
            args.source_layers,
            "--target-layer",
            str(args.target_layer),
            "--dim-batch",
            str(dim_batch),
            "--out-dir",
            str(coffee_out),
        ]
        if args.fresh_v2:
            coffee.append("--no-resume")
        commands.append(("01_v2_fit_and_eval", coffee))

    if not args.skip_precoffee:
        precoffee = [
            python,
            str(SCRIPTS_DIR / "run_othello_precoffee.py"),
            "--device",
            args.device,
            *checkpoint,
            "--lens",
            str(lens_path),
            "--out-dir",
            args.precoffee_out,
        ]
        if args.quick:
            precoffee.append("--quick")
        commands.append(("02_precoffee_three_experiments", precoffee))

    if not args.skip_closure:
        closure = [
            python,
            str(EXPERIMENT_DIR / "experiment.py"),
            "--device",
            args.device,
            *checkpoint,
            "--lens",
            str(lens_path),
        ]
        if args.quick:
            closure.append("--quick")
        commands.append(("03_workspace_closure", closure))
    return commands


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--checkpoint",
        help="Local synthetic_model.pth; omit to download it on the GPU node",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for child stages (default: this interpreter)",
    )
    parser.add_argument("--coffee-out", default="out/coffee_v2")
    parser.add_argument("--precoffee-out", default="out/precoffee")
    parser.add_argument("--n-prompts", type=int, default=100)
    parser.add_argument("--n-games", type=int, default=100)
    parser.add_argument("--source-layers", default="0,1,2,3,4,5,6")
    parser.add_argument("--target-layer", type=int, default=7)
    parser.add_argument("--dim-batch", type=int, default=128)
    parser.add_argument(
        "--fresh-v2",
        action="store_true",
        help="Ignore any partial v2 fit checkpoint instead of resuming it",
    )
    parser.add_argument("--skip-v2", action="store_true")
    parser.add_argument("--skip-precoffee", action="store_true")
    parser.add_argument("--skip-closure", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run small smoke-test sizes for all stages",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print every command without launching it",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    commands = _commands(args)
    if not commands:
        raise SystemExit("all stages were skipped")

    manifest = {
        "pipeline": "othello-v2-and-all-workspace-experiments",
        "started_at_unix": time.time(),
        "repository": str(REPO_ROOT),
        "quick": args.quick,
        "dry_run": args.dry_run,
        "stages": [],
    }
    if not args.dry_run:
        _write_manifest(manifest)
    for name, command in commands:
        _run_stage(name, command, manifest, dry_run=args.dry_run)
    if not args.dry_run:
        manifest["ended_at_unix"] = time.time()
        manifest["elapsed_seconds"] = manifest["ended_at_unix"] - manifest["started_at_unix"]
        manifest["status"] = "complete"
        _write_manifest(manifest)
        print(f"\nAll stages complete. Manifest: {MANIFEST_PATH}", flush=True)


if __name__ == "__main__":
    main()
