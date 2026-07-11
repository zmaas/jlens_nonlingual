# Jacobian Lens on non-language models

Minimal adapters and experiments for applying Anthropic's Jacobian Lens to
non-language models. Phase 1 is an OthelloGPT coffee demo; real Evo2 work is
intentionally deferred until that smoke test passes.

Anthropic's implementation is vendored at `vendor/jacobian-lens` and retains
its Apache-2.0 license. The algorithm is reused rather than reimplemented.

See [`docs/othello_evo2_jlens.md`](docs/othello_evo2_jlens.md) for the cloud
GPU command, local-checkpoint option, fallbacks, outputs, and interpretation.
The follow-up legality, causal-steering, and workspace-split experiments are
documented in
[`docs/precoffee_othello_workspace.md`](docs/precoffee_othello_workspace.md).

Package management uses `uv`. This checkout intentionally does not contain
model weights or an installed environment.
