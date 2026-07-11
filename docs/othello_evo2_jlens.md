# OthelloGPT Jacobian Lens

Phase 1 is a minimal non-language smoke test of Anthropic's average-Jacobian
lens. The fitting and application algorithms live in the vendored upstream at
`vendor/jacobian-lens`; this repository adds a TransformerLens adapter, legal
synthetic Othello prompts, evaluation, and a compact report.

## Cloud GPU setup

No model weights or Python packages are vendored. On the GPU machine:

```bash
uv sync
uv run python scripts/run_othello_coffee_demo.py \
  --device cuda \
  --n-prompts 10 \
  --dim-batch 128 \
  --out-dir out/coffee
```

The first run downloads `synthetic_model.pth` from
`NeelNanda/Othello-GPT-Transformer-Lens`. To control or pre-stage that download,
pass `--checkpoint /path/to/synthetic_model.pth` to the wrapper or either
underlying script.

Low-memory CUDA fallback:

```bash
uv run python scripts/run_othello_coffee_demo.py \
  --device cuda --n-prompts 3 --source-layers 5,6 \
  --target-layer 7 --dim-batch 32 --out-dir out/coffee_fast
```

CPU fallback:

```bash
uv run python scripts/run_othello_coffee_demo.py \
  --device cpu --n-prompts 1 --source-layers 6 \
  --target-layer 7 --dim-batch 8 --out-dir out/coffee_cpu
```

The main run creates `othello_jlens.pt`, `othello_eval.json`, and
`othello_summary.md`. Fitting is resumable through `othello_fit_ckpt.pt`.

OthelloGPT encodes the 60 playable squares as tokens `1..60`; token `0` is
unused. A pass changes the active player but does not add a sequence token.
Checkpoint and lens metadata record this encoding, and the scripts refuse to
resume or evaluate artifacts created with a different or unknown encoding.

## Interpretation

This direct J-lens decodes move tokens, not board-state labels. Legal-move
enrichment is suggestive only; board occupancy requires a probe/template
extension.

The evaluation reports held-out next-move pass@5 and median rank for J-lens,
the ordinary logit lens, and final logits. These are move-token disposition
metrics, not evidence that a layer directly represents board occupancy.
