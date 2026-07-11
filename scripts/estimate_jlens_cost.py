#!/usr/bin/env python3
"""Print storage and backward-pass counts for a Jacobian Lens fit."""

from __future__ import annotations

import argparse
import math


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-source-layers", type=int, default=7)
    parser.add_argument("--n-prompts", type=int, default=10)
    parser.add_argument("--dim-batch", type=int, default=128)
    parser.add_argument("--storage-dtype-bytes", type=int, default=2)
    args = parser.parse_args()
    matrices = args.n_source_layers * args.d_model * args.d_model
    passes_per_prompt = math.ceil(args.d_model / args.dim_batch)
    print(f"matrix elements: {matrices:,}")
    print(f"saved lens storage: {matrices * args.storage_dtype_bytes / 2**20:.2f} MiB")
    print(f"fit accumulator storage (fp32): {matrices * 4 / 2**20:.2f} MiB")
    print(f"backward passes per prompt: {passes_per_prompt}")
    print(f"total backward passes: {passes_per_prompt * args.n_prompts}")
    print("forward passes:", args.n_prompts)


if __name__ == "__main__":
    main()
