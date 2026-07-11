# Othello workspace closure experiment

## Question

Does OthelloGPT expose a compact, causally useful move workspace, rather than a
60-dimensional subspace that merely inherits information from the output
vocabulary?

## Design

The experiment runs four linked tests against the existing OthelloGPT checkpoint
and fitted J-lens:

1. **Relative-state probes.** Decode absolute board state, board state relative
   to the player to move, player identity, legal moves, and the sampled next
   move from the full residual stream, the full linear J-span, its orthogonal
   remainder, and a matched random span. Board scores use macro balanced
   accuracy across empty/current-player/opponent labels.
2. **Rank sweep.** Keep or remove the leading 1, 2, 4, 8, 16, 32, 59, or 60
   singular directions of the linear J-space and measure causal changes in move
   behavior at every layer.
3. **Sparse nonnegative sweep.** Approximate the activation-dependent J-lens
   cone with nonnegative gradient pursuit at several support sizes, then test
   both reconstruction quality and causal keep/remove interventions. This is an
   explicit local approximation; the vendored public J-lens repository does not
   provide Anthropic's gradient-pursuit implementation.
4. **Weak-mode diagnostic.** Inspect the final right-singular vector, its
   alignment with the uniform-token direction, and control spectra using the
   raw and centered move unembeddings.

The causal primary outcomes are legal probability mass and legal precision at
5. The deterministic next-token metric is named **sampled-target top-5
inclusion**, not pass@5.

## Status

Implementation complete; GPU execution is pending. Run `experiment.py` from the
repository root. The script writes `data/results.json`, transparent SVG figures,
runtime logs, and replaces this section with an empirical summary.

## Run

```bash
python experiments/2026-07-11_15-14-othello-workspace-closure/experiment.py \
  --device cuda \
  --checkpoint /path/to/synthetic_model.pth \
  --lens out/coffee_v2/othello_jlens.pt
```

For a short validation run, add `--quick`.
