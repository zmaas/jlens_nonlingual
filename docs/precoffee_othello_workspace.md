# Pre-coffee Othello workspace experiments

These experiments reuse the corrected 100-prompt lens in
`out/coffee_v2/othello_jlens.pt`; they do not refit Jacobians.

## Run

Full run:

```bash
uv run python scripts/run_othello_precoffee.py \
  --device cuda \
  --lens out/coffee_v2/othello_jlens.pt \
  --out-dir out/precoffee
```

Fast end-to-end validation before the full run:

```bash
uv run python scripts/run_othello_precoffee.py \
  --device cuda \
  --lens out/coffee_v2/othello_jlens.pt \
  --out-dir out/precoffee_quick \
  --quick
```

Pass `--checkpoint /path/to/synthetic_model.pth` to avoid a Hub lookup.

After all three experiments, the wrapper writes a combined coffee brief and a
single nested machine-readable result:

```text
out/precoffee/precoffee_summary.md
out/precoffee/results.json
```

## Experiment 1: legal-move emergence

`eval_othello_legality.py` measures legal precision@5, recall@5, legal
probability mass, any-legal@5, unused-token rate, and sampled-target top-5 inclusion for
J-lens, logit lens, and final logits. Confidence intervals resample whole games;
paired J-minus-logit results preserve position-level pairing within games.

Artifacts:

```text
out/precoffee/legal/legal_eval.json
out/precoffee/legal/legal_summary.md
out/precoffee/legal/legal_precision_by_layer.svg
out/precoffee/legal/legal_mass_by_layer.svg
```

## Experiment 2: causal move steering

`intervene_othello_jspace.py` adds norm-scaled directions to a single residual
position and measures the downstream target-logit, rank, top-five, legal-mass,
and output-KL effects. It compares:

- target move J-lens direction;
- target move unembedding/logit direction;
- norm-matched random direction;
- a different move's J-lens direction.

Legal and illegal target squares are evaluated at three intervention strengths.

Artifacts:

```text
out/precoffee/steering/steering_results.json
out/precoffee/steering/steering_summary.md
out/precoffee/steering/target_logit_effect_by_layer.svg
```

## Experiment 3: J-space versus orthogonal computation

`analyze_othello_workspace_split.py` constructs the span of tokens 1–60's
J-lens directions at every layer. Ridge probes compare full activations,
J-space coordinates, the orthogonal residual, and a matched random projection
on board state, legal moves, player identity, and sampled next move.

This linear **J-span** is a tractable subspace proxy. Anthropic's formal J-space
is instead the set of sparse nonnegative combinations of J-lens vectors—a union
of cones rather than the entire linear span. Any positive result here motivates
the closer sparse-decomposition analysis; it does not establish it by itself.

The causal portion removes J-space, the ordinary unembedding span, or a matched
random span at all positions, and also tests retaining only J-space. It reports
legal behavior and output-distribution damage with game-level bootstrap
intervals.

Artifacts:

```text
out/precoffee/workspace/workspace_split.json
out/precoffee/workspace/workspace_split_summary.md
out/precoffee/workspace/probe_accuracy_by_component.svg
out/precoffee/workspace/jspace_variance_by_layer.svg
out/precoffee/workspace/ablation_effect_by_layer.svg
```

## Interpretation boundary

A workspace-like result requires more than readable next-move tokens. The
strongest pattern would combine:

1. legal-action content concentrated in a small J-space component;
2. causal, target-specific control from writing a move direction;
3. disproportionate action impairment after J-space removal relative to a
   matched random subspace;
4. detailed board state remaining more available in the orthogonal component.

Failure to find this dissociation is informative: it would support a model with
progressively refined output representations but no clearly privileged global
workspace.
