You are working on a minimal Jacobian Lens / J-Space implementation path.

Context:
We want to inspect Anthropic’s new J-Space / Jacobian Lens method and build a minimal reimplementation path that reuses existing code wherever possible. The first priority is a same-day OthelloGPT smoke demo for discussion with an Anthropic researcher. Evo2 transfer is second priority and should be staged by GPU size.

Primary decision:
Do not clean-room reimplement the Jacobian Lens algorithm. Reuse Anthropic’s public `anthropics/jacobian-lens` implementation as the core. Add adapters, scripts, tests, and documentation only.

Core method:
The Jacobian Lens estimates an average linear transport map from an intermediate residual stream to a final residual stream, then decodes through the model’s own unembedding. In practice, for source layer `l`, fit an average Jacobian `J_l`, then decode approximately with:

  lens_l(h_l) = unembed(J_l @ h_l)

Keep Anthropic’s design choices:
- Use `jlens.fit(...)` for fitting.
- Use `JacobianLens.apply(...)` for applying.
- Use existing checkpointing and merge support.
- Keep the core algorithm mostly unchanged.
- Adapt models to the `LensModel` abstraction instead of rewriting fitting logic.

Important interpretation caveat:
Direct J-lens readouts are output-token-level. For OthelloGPT, the readout is move-token disposition, not board-state labels. For Evo2, the readout is nucleotide/token-level disposition, not direct biological concepts. Stronger semantic/board/biology claims require later template-lens, probe-bridge, or concept-lens extensions.

Hard priority order:
1. OthelloGPT coffee demo.
2. OthelloGPT better eval.
3. Evo2 adapter/fake tests.
4. Evo2 7B single-layer real smoke.
5. Evo2 7B selected-layer first pass.
6. Anything larger than Evo2 7B is out of scope for v0.

Do not attempt Evo2 real fitting before the Othello coffee demo passes.

Relevant upstreams:
- Anthropic Jacobian Lens repo: https://github.com/anthropics/jacobian-lens
- OthelloGPT TransformerLens demo/checkpoint: `NeelNanda/Othello-GPT-Transformer-Lens/synthetic_model.pth`
- Evo2 repo: https://github.com/ArcInstitute/evo2

Known useful model facts:
- OthelloGPT TransformerLens demo model:
  - n_layers = 8
  - d_model = 512
  - d_head = 64
  - n_heads = 8
  - d_mlp = 2048
  - d_vocab = 61
  - n_ctx = 59
  - activation = GELU
  - normalization = LNPre
- Evo2 7B:
  - target only Evo2 7B variants for v0
  - approximately 32 layers
  - hidden size / d_model = 4096
  - vocab size = 512
  - start with `evo2_7b_base`
  - start with `use_kernels=False` for fitting/backward
  - larger Evo2 models are out of scope for v0 because of FP8 / Transformer Engine / Hopper complexity

Repository changes to make:

Add files:
- `jlens/adapters/__init__.py`
- `jlens/adapters/transformer_lens.py`
- `jlens/adapters/evo2.py`
- `scripts/run_othello_coffee_demo.py`
- `scripts/fit_othello_jlens.py`
- `scripts/eval_othello_jlens.py`
- `scripts/check_evo2_adapter.py`
- `scripts/fit_evo2_jlens.py`
- `scripts/apply_evo2_jlens.py`
- `scripts/merge_jlenses.py`
- `scripts/estimate_jlens_cost.py`
- `tests/test_transformer_lens_adapter.py`
- `tests/test_evo2_adapter_tiny.py`
- optional: `tests/test_othello_smoke.py`, marked slow/integration
- `docs/othello_evo2_jlens.md`

Small core change:
Generalize prompt type hints in the Jacobian Lens fitting API.

Current code may assume `Sequence[str]`. Change type hints/docstrings to accept arbitrary prompt-like inputs:
- `Sequence[Any]`
- or define `PromptLike = Any`

Do not change runtime behavior except to allow adapters to handle:
- strings
- integer token lists
- integer tensors

All prompts should still pass through:

  model.encode(prompt, max_length=...)

Existing examples and tests should still work.

================================================================================
GPU-STAGED IMPLEMENTATION SPEC
================================================================================

Stage 0 — CPU or any dev machine
Purpose:
- Repository setup.
- Fake tiny model tests.
- Adapter skeletons.
- No real model weights required.

Tasks:
- Vendor or depend on `anthropics/jacobian-lens`.
- Add `TransformerLensLensModel`.
- Add `Evo2LensModel`, but test against a fake tiny Evo2-like module.
- Add `estimate_jlens_cost.py`.

Commands:
  pytest tests/test_transformer_lens_adapter.py
  pytest tests/test_evo2_adapter_tiny.py

Definition of done:
- Tests pass without downloading Evo2.
- Fitting works on a fake 2-layer model.
- `estimate_jlens_cost.py` prints matrix storage and backward-pass counts.
- No GPU required.

Expected cost:
- CPU only.

--------------------------------------------------------------------------------

Stage 1 — coffee demo, 8–16 GB GPU preferred
Purpose:
- Produce a same-day OthelloGPT demo.
- Avoid Evo2 entirely.
- Show J-lens top-k move-token readouts by layer.

Target hardware:
- Any CUDA GPU with 8–16 GB VRAM.
- CPU fallback allowed with fewer prompts/layers.

Implementation tasks:
- Finish `TransformerLensLensModel`.
- Add `scripts/fit_othello_jlens.py`.
- Add `scripts/eval_othello_jlens.py`.
- Add `scripts/run_othello_coffee_demo.py` as one-command wrapper.
- Output compact markdown summary at `out/coffee/othello_summary.md`.

Coffee-demo defaults:
  model: OthelloGPT synthetic TransformerLens checkpoint
  source_layers: 0,1,2,3,4,5,6
  target_layer: 7
  n_prompts: 10
  max_seq_len: 59
  dim_batch: 128
  positions: last token and 3 midgame positions
  eval k: 5
  baseline: logit lens and final logits

Primary command:
  python scripts/run_othello_coffee_demo.py \
    --device cuda \
    --n-prompts 10 \
    --dim-batch 128 \
    --out-dir out/coffee

Equivalent explicit commands:
  python scripts/fit_othello_jlens.py \
    --device cuda \
    --n-prompts 10 \
    --max-seq-len 59 \
    --source-layers 0,1,2,3,4,5,6 \
    --target-layer 7 \
    --dim-batch 128 \
    --out out/coffee/othello_jlens.pt \
    --checkpoint-path out/coffee/othello_fit_ckpt.pt

  python scripts/eval_othello_jlens.py \
    --device cuda \
    --lens out/coffee/othello_jlens.pt \
    --n-games 25 \
    --k 5 \
    --out out/coffee/othello_eval.json \
    --markdown out/coffee/othello_summary.md

Fallback command if CUDA memory or time is bad:
  python scripts/run_othello_coffee_demo.py \
    --device cuda \
    --n-prompts 3 \
    --source-layers 5,6 \
    --target-layer 7 \
    --dim-batch 32 \
    --out-dir out/coffee_fast

CPU fallback:
  python scripts/run_othello_coffee_demo.py \
    --device cpu \
    --n-prompts 1 \
    --source-layers 6 \
    --target-layer 7 \
    --dim-batch 8 \
    --out-dir out/coffee_cpu

Definition of done:
- `out/coffee/othello_jlens.pt` exists.
- `out/coffee/othello_eval.json` exists.
- `out/coffee/othello_summary.md` exists and contains:
  - model/config summary;
  - source/target layers;
  - number of prompts;
  - next-move pass@5 by layer;
  - median next-move rank by layer;
  - logit-lens baseline;
  - 3 example prefixes with per-layer top-5 decoded move tokens.
- Summary must include this exact caveat:
  “This direct J-lens decodes move tokens, not board-state labels. Legal-move enrichment is suggestive only; board occupancy requires a probe/template extension.”

Coffee-demo talking points to include in markdown:
- We reused Anthropic’s average-Jacobian lens design unchanged.
- The new work is an adapter: TransformerLens residual streams -> J-lens API -> Othello move-token unembedding.
- This is the smallest non-language sanity check before Evo2.
- The direct readout should be interpreted as “what move-token futures this activation linearly transports to,” not “the board state is decoded.”
- Next step is a board-state template/probe lens if move-token readout is clean.

--------------------------------------------------------------------------------

Stage 2 — better Othello run, 16–24 GB GPU
Purpose:
- Make the Othello result less toyish.
- Still do not touch Evo2 unless Stage 1 passes.

Target hardware:
- 16–24 GB GPU.

Defaults:
  n_prompts: 100
  source_layers: 0..6
  target_layer: 7
  dim_batch: 128 or 256
  n_eval_games: 100–1000
  legal-move engine: optional but preferred

Command:
  python scripts/run_othello_coffee_demo.py \
    --device cuda \
    --n-prompts 100 \
    --dim-batch 256 \
    --n-games 250 \
    --out-dir out/othello_100prompt

Additional metrics:
- `legal_precision_at_k_by_layer`, if legal-move engine available.
- `legal_recall_at_k_by_layer`, if legal-move engine available.
- `j_lens_minus_logit_lens_pass_at_k`.
- `j_lens_minus_logit_lens_median_rank`.

Definition of done:
- J-lens, logit lens, and final-logit baselines are all in one JSON.
- Markdown summary has a small table suitable for sharing.
- No board-state claims.

--------------------------------------------------------------------------------

Stage 3 — Evo2 adapter plumbing, 24 GB GPU optional
Purpose:
- Test Evo2 adapter shape/device logic without fitting a real Evo2 J-lens.

Target hardware:
- CPU is enough for fake tests.
- 24 GB GPU can be used to try loading Evo2 7B, but this is optional.

Tasks:
- Finish `Evo2LensModel`.
- Add `scripts/check_evo2_adapter.py`.
- Add `scripts/fit_evo2_jlens.py`, but gate real fitting behind `--allow-real-fit`.
- Add `scripts/apply_evo2_jlens.py`.
- Add `tests/test_evo2_adapter_tiny.py`.

Fake tiny test:
  pytest tests/test_evo2_adapter_tiny.py

Optional real-load check:
  python scripts/check_evo2_adapter.py \
    --model-name evo2_7b_base \
    --max-seq-len 64 \
    --device cuda \
    --no-fit

Definition of done:
- Fake Evo2 adapter test passes.
- Optional real-load check can tokenize DNA, run a forward pass, and unembed to vocab dimension 512.
- No Jacobian fitting required.

Important:
- Use `use_kernels=False` for fitting-related code until backward compatibility is confirmed.
- Do not use hosted NIM/API endpoints for J-lens fitting; we need local autograd.

--------------------------------------------------------------------------------

Stage 4 — Evo2 7B tiny real fit, 48 GB preferred, 24 GB risky
Purpose:
- Check whether a real Evo2 backward pass works at all.

Target hardware:
- 48 GB preferred.
- 24 GB may work only with very short sequence, one prompt, one source layer, `dim_batch=1`.
- Do not assume 24 GB is stable.

Defaults:
  model: evo2_7b_base
  source_layers: 30
  target_layer: 31
  n_prompts: 1
  max_seq_len: 64
  dim_batch: 1
  skip_first: 16
  use_kernels: false

Command:
  python scripts/fit_evo2_jlens.py \
    --model-name evo2_7b_base \
    --source-layers 30 \
    --target-layer 31 \
    --n-prompts 1 \
    --max-seq-len 64 \
    --dim-batch 1 \
    --skip-first 16 \
    --use-kernels false \
    --out out/evo2_tiny/lens.pt \
    --checkpoint-path out/evo2_tiny/ckpt.pt \
    --allow-real-fit

Definition of done:
- One prompt completes.
- Lens saves.
- `apply_evo2_jlens.py` prints top-k token-level readout.
- Output is explicitly labeled nucleotide/token-level readout.

Failure policy:
- If real Evo2 backward fails due to kernels or Vortex internals, preserve the adapter and fake tests.
- Do not spend time patching 20B/40B paths.
- File a clear note:
  “Evo2 real fit blocked on local backward through Vortex path.”

--------------------------------------------------------------------------------

Stage 5 — Evo2 7B first useful run, 80 GB GPU
Purpose:
- Produce first actual Evo2 token-level J-lens readout.

Target hardware:
- A100 80 GB or H100 80 GB.

Defaults, first pass:
  model: evo2_7b_base
  source_layers: 30
  target_layer: 31
  n_prompts: 5
  max_seq_len: 128
  dim_batch: 1
  skip_first: 16
  use_kernels: false

Command:
  python scripts/fit_evo2_jlens.py \
    --model-name evo2_7b_base \
    --source-layers 30 \
    --target-layer 31 \
    --n-prompts 5 \
    --max-seq-len 128 \
    --dim-batch 1 \
    --skip-first 16 \
    --use-kernels false \
    --out out/evo2_7b_stage5/lens_L30.pt \
    --checkpoint-path out/evo2_7b_stage5/ckpt.pt \
    --allow-real-fit

Then apply:
  python scripts/apply_evo2_jlens.py \
    --model-name evo2_7b_base \
    --lens out/evo2_7b_stage5/lens_L30.pt \
    --sequence ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT \
    --positions -2 \
    --top-k 10 \
    --out out/evo2_7b_stage5/readout.json

Definition of done:
- Token-level top-k readouts are produced for layer 30.
- Final logits and logit-lens baseline are included.
- Output explains that Evo2 readouts are token-level/nucleotide-level, not biological concept readouts.

--------------------------------------------------------------------------------

Stage 6 — Evo2 7B selected-layer run, 80 GB GPU, overnight-scale
Purpose:
- Expand from one layer to selected layers.

Target hardware:
- A100/H100 80 GB.

Defaults:
  source_layers: 24,30 first
  then: 16,24,30
  then: 8,16,24,30
  target_layer: 31
  n_prompts: 10–20
  max_seq_len: 128
  dim_batch: 1 initially; test 2/4/8 only after profiling

Command:
  python scripts/fit_evo2_jlens.py \
    --model-name evo2_7b_base \
    --source-layers 24,30 \
    --target-layer 31 \
    --n-prompts 10 \
    --max-seq-len 128 \
    --dim-batch 1 \
    --skip-first 16 \
    --use-kernels false \
    --out out/evo2_7b_stage6/lens_L24_L30.pt \
    --checkpoint-path out/evo2_7b_stage6/ckpt.pt \
    --allow-real-fit

Definition of done:
- Selected-layer lens saves.
- Apply script can compare J-lens vs logit lens vs final logits.
- No semantic biology claims beyond output-token/nucleotide readouts.

--------------------------------------------------------------------------------

Stage 7 — shard and merge, multiple 80 GB GPUs or repeated jobs
Purpose:
- Use Anthropic’s merge design for prompt shards.

Target hardware:
- Multiple 80 GB GPUs, or repeated single-GPU jobs.

Shard command:
  python scripts/fit_evo2_jlens.py \
    --model-name evo2_7b_base \
    --source-layers 8,16,24,30 \
    --target-layer 31 \
    --prompts-file data/evo2_prompts_shard_00.txt \
    --max-seq-len 128 \
    --dim-batch 1 \
    --out out/evo2_shards/lens_shard_00.pt \
    --checkpoint-path out/evo2_shards/ckpt_shard_00.pt \
    --allow-real-fit

Merge command:
  python scripts/merge_jlenses.py \
    --inputs out/evo2_shards/lens_shard_*.pt \
    --out out/evo2_7b_merged/lens.pt

Definition of done:
- Merged lens has correct `n_prompts`.
- Per-shard metadata is retained.
- Apply script works on merged lens.

================================================================================
ADAPTER SPEC
================================================================================

1. `TransformerLensLensModel`

File:
  `jlens/adapters/transformer_lens.py`

Purpose:
Wrap a `transformer_lens.HookedTransformer` so it satisfies Anthropic Jacobian Lens `LensModel`.

Constructor:
  TransformerLensLensModel(
      tl_model,
      id_to_label: dict[int, str] | None = None,
      device: str | torch.device | None = None,
  )

Required attributes:
- `self.model = tl_model`
- `self.n_layers = tl_model.cfg.n_layers`
- `self.d_model = tl_model.cfg.d_model`
- `self.layers = tl_model.blocks`
- `self.tokenizer = TransformerLensTokenizerShim(...)`

`encode(prompt, max_length)`:
- If `prompt` is a `torch.Tensor`:
  - ensure shape `[1, seq]`;
  - dtype `torch.long`;
  - truncate to `max_length`;
  - move to model device.
- If `prompt` is `list[int]` or `tuple[int]`:
  - convert to tensor shape `[1, seq]`;
  - truncate;
  - move to model device.
- If `prompt` is `str`:
  - first try parsing comma/space-separated integers;
  - if that fails and underlying model has normal tokenizer support, use that;
  - for Othello, integer token sequences are canonical.
- Return token tensor shape `[1, seq]`.

`forward(input_ids)`:
- Call underlying HookedTransformer forward normally.
- Do not wrap in `torch.no_grad()`.
- Return value can be model logits. `ActivationRecorder` mainly needs hooks to fire.

`unembed(residual)`:
- Apply final layer norm if present.
- Then apply unembedding.
- Return logits shape `[..., d_vocab]`.

Final norm handling:
- Prefer using TransformerLens utilities if clean.
- Otherwise:
  - if model has `ln_final`, call it before unembedding;
  - then call `model.unembed`.

Tokenizer shim:
- Implement `decode(ids)`.
- For Othello default mapping: `M{token_id}`.
- Allow optional `id_to_label` so move IDs can later be board coordinates.

Important:
- Set model to eval mode.
- Freeze parameters with `requires_grad_(False)`.
- Do not disable activation gradients.
- Do not import Evo2 in this file.

--------------------------------------------------------------------------------

2. `Evo2LensModel`

File:
  `jlens/adapters/evo2.py`

Purpose:
Wrap an Evo2/Vortex model so it satisfies Anthropic Jacobian Lens `LensModel`.

Constructor:
  Evo2LensModel(
      evo_or_model_name,
      use_kernels: bool = False,
      device: str | torch.device | None = None,
  )

Accept either:
- an `evo2.Evo2` instance; or
- a model name/local path, e.g. `evo2_7b_base`.

Store:
- `self.evo = evo2_model`
- `self.model = evo2_model.model`
- `self.layers = self.model.blocks`
- `self.n_layers = self.model.config.num_layers`
- `self.d_model = self.model.config.hidden_size`
- `self.tokenizer = Evo2TokenizerShim(evo2_model.tokenizer)`

`encode(prompt, max_length)`:
- If prompt is a DNA string:
  - call Evo2 tokenizer.
- If prompt is already token IDs:
  - accept list/tuple/tensor.
- Truncate to `max_length`.
- Convert to `torch.long` tensor shape `[1, seq]`.
- Move to first model device, preferably:
  - `self.model.block_idx_to_device[0]`, if available;
  - else next parameter device.

`forward(input_ids)`:
- Call the underlying local model directly:
  - `self.model.forward(input_ids)`
- Do not call hosted endpoints.
- Do not call wrappers that use `torch.no_grad()`.
- Do not use cached generation or `inference_params_dict`.
- Preserve autograd.

`unembed(residual)`:
- Move residual to the unembedding/norm device if needed.
- Apply `self.model.norm` if not None.
- Apply `self.model.unembed`.
- Return logits.

Tokenizer shim:
- Implement `decode(ids)`.
- Use Evo2 tokenizer decode if possible.
- Fallback to `str(int(token_id))`.

Important Evo2 settings:
- Start with `use_kernels=False`.
- Start with 7B only.
- Start with short `max_seq_len`, default 64 or 128.
- Use selected layers only, not all layers.
- Recommended first real source layers:
  - `[30]`
  - then `[24, 30]`
  - then `[16, 24, 30]`
  - then `[8, 16, 24, 30]`
- Target layer:
  - `31`
- Recommended first prompts:
  - 1 for smoke
  - 5 for first useful run
  - 10–20 for selected-layer run
- Recommended first `dim_batch`:
  - `1`
  - raise to 2/4/8 only after memory profiling

================================================================================
SCRIPT SPEC
================================================================================

1. `scripts/run_othello_coffee_demo.py`

Purpose:
One-command OthelloGPT demo. Must not import Evo2.

CLI:
- `--device`, default `cuda` if available else `cpu`
- `--n-prompts`, default `10`
- `--n-games`, default `25`
- `--source-layers`, default `0,1,2,3,4,5,6`
- `--target-layer`, default `7`
- `--dim-batch`, default `128`
- `--max-seq-len`, default `59`
- `--k`, default `5`
- `--out-dir`, default `out/coffee`
- `--checkpoint-path`, default `{out_dir}/othello_fit_ckpt.pt`

Behavior:
- Ensure output directory exists.
- Load OthelloGPT.
- Fit J-lens.
- Evaluate J-lens.
- Write:
  - `{out_dir}/othello_jlens.pt`
  - `{out_dir}/othello_eval.json`
  - `{out_dir}/othello_summary.md`
- Print path to markdown summary.

Must be robust:
- If Othello legal engine is unavailable, still run next-token metrics.
- If CUDA OOM occurs, print fallback command using fewer prompts/layers.
- Do not crash because Evo2 is not installed.

--------------------------------------------------------------------------------

2. `scripts/fit_othello_jlens.py`

CLI:
- `--out`
- `--n-prompts`
- `--dim-batch`
- `--max-seq-len`
- `--skip-first`
- `--source-layers`
- `--target-layer`
- `--device`
- `--checkpoint-path`

Behavior:
- Build/load OthelloGPT via TransformerLens.
- Use synthetic checkpoint `NeelNanda/Othello-GPT-Transformer-Lens/synthetic_model.pth`.
- Generate or load Othello move-token sequences.
- Wrap with `TransformerLensLensModel`.
- Fit selected source layers to target layer.
- Save lens.

Acceptance:
- Produces a valid `JacobianLens`.
- For coffee defaults:
  - `d_model = 512`
  - source layers `[0,1,2,3,4,5,6]`
  - target layer `7`
  - `n_prompts > 0`
- Does not require board-state probes.

--------------------------------------------------------------------------------

3. `scripts/eval_othello_jlens.py`

CLI:
- `--lens`
- `--n-games`
- `--k`
- `--device`
- `--out`
- `--markdown`
- optional `--source-layers`
- optional `--positions`

Metrics:
- `next_move_pass_at_k_by_layer`
- `next_move_rank_by_layer`
- `median_next_move_rank_by_layer`
- `legal_precision_at_k_by_layer`, if legal-move computation is available
- `legal_recall_at_k_by_layer`, if legal-move computation is available
- `logit_lens_baseline`
- `final_logits_baseline`

Example outputs:
For 3 example prefixes, include:
- prefix length
- true next move token
- final model top-5
- logit lens top-5 at layers 3,5,6 where available
- J-lens top-5 at layers 3,5,6 where available

Markdown structure:
  # OthelloGPT Jacobian Lens smoke test

  ## Setup
  - Model: OthelloGPT synthetic TransformerLens checkpoint
  - Layers: source 0–6, target 7
  - Prompts used: N
  - Method: average Jacobian transport from source residual stream to final residual stream, decoded with model unembedding

  ## Main result
  Table:
  layer | next-move pass@5 | median next-move rank | logit-lens pass@5 | delta

  ## Three examples
  For each example:
  - prefix length
  - true next move token
  - final model top-5
  - logit lens top-5 at layer 3/5/6
  - J-lens top-5 at layer 3/5/6

  ## Interpretation
  This is a readout sanity check. It shows whether J-lens exposes move-token dispositions earlier or more cleanly than logit lens. It does not directly decode Othello board state.

  ## Next steps
  - Add legal-move precision/recall.
  - Add board-state probe/template bridge.
  - Port adapter pattern to Evo2 7B.

Required exact caveat:
  “This direct J-lens decodes move tokens, not board-state labels. Legal-move enrichment is suggestive only; board occupancy requires a probe/template extension.”

Acceptance:
- JSON output includes per-layer metrics.
- Markdown output is human-readable.
- At least one mid/late layer should beat random next-move rank in normal runs, but tests should not be brittle on this.
- No board-state claims.

--------------------------------------------------------------------------------

4. `scripts/check_evo2_adapter.py`

Purpose:
Check Evo2 loading/forward/unembedding without fitting a Jacobian Lens.

CLI:
- `--model-name`, default `evo2_7b_base`
- `--max-seq-len`, default `64`
- `--device`, default `cuda`
- `--use-kernels`, default false
- `--no-fit`, default true

Behavior:
- Load Evo2.
- Wrap with `Evo2LensModel`.
- Tokenize a small DNA string.
- Run forward.
- Run unembed.
- Print shape summary:
  - token shape
  - hidden shape if available
  - logits shape
  - vocab dimension
  - devices touched

Acceptance:
- Does not fit.
- Does not call hosted endpoints.
- Useful for Stage 3.

--------------------------------------------------------------------------------

5. `scripts/fit_evo2_jlens.py`

CLI:
- `--model-name`, default `evo2_7b_base`
- `--out`
- `--source-layers`
- `--target-layer`
- `--prompts-file`
- `--n-prompts`
- `--max-seq-len`
- `--dim-batch`
- `--skip-first`
- `--checkpoint-path`
- `--use-kernels`, default false
- `--allow-real-fit`, required for real fitting

Behavior:
- Refuse to run unless `--allow-real-fit` is passed.
- Load Evo2 7B base by default.
- Wrap with `Evo2LensModel`.
- Load DNA prompts from file if provided.
- If no prompt file, use tiny built-in synthetic DNA prompts over `ACGT`.
- Fit selected source layers only.
- Save lens.
- Log:
  - model name
  - source layers
  - target layer
  - n prompts
  - max seq len
  - dim batch
  - estimated backward passes
  - estimated matrix storage
  - elapsed time per prompt if easy
  - memory notes if available

Acceptance:
- Stage 4 command can attempt one prompt, one source layer.
- Saves a lens if backward works.
- If backward fails, error should be informative.

--------------------------------------------------------------------------------

6. `scripts/apply_evo2_jlens.py`

CLI:
- `--model-name`
- `--lens`
- `--sequence`
- `--positions`
- `--top-k`
- `--out`
- `--use-kernels`, default false

Behavior:
- Load Evo2.
- Load saved lens.
- Apply lens at selected positions.
- Also compute:
  - final model top-k logits at same position
  - logit-lens baseline top-k if supported
- Write human-readable JSON.

JSON should include, for each layer and position:
- token IDs
- decoded token strings
- raw J-lens logits
- final model top-k
- logit-lens baseline top-k
- caveat string:
  “This is a token-level Evo2 J-lens readout, not a biological concept readout.”

Acceptance:
- Can run against saved or tiny test lens.
- Does not make biological semantic claims.

--------------------------------------------------------------------------------

7. `scripts/merge_jlenses.py`

CLI:
- `--inputs`
- `--out`

Behavior:
- Load multiple JacobianLens files.
- Merge using Anthropic’s existing merge support.
- Save merged lens.
- Preserve metadata if practical:
  - input paths
  - n_prompts per shard
  - total n_prompts
  - source layers
  - target layer

Acceptance:
- Merged lens applies successfully.
- Correct total prompt count.

--------------------------------------------------------------------------------

8. `scripts/estimate_jlens_cost.py`

Purpose:
Make GPU staging explicit before jobs start.

CLI:
- `--model-kind othello|evo2_7b|custom`
- `--d-model`
- `--n-source-layers`
- `--n-prompts`
- `--dim-batch`
- `--max-seq-len`
- optional `--gpu-vram-gb`

Defaults:
- If `--model-kind othello`:
  - d_model = 512 unless overridden
- If `--model-kind evo2_7b`:
  - d_model = 4096 unless overridden

Print:
- estimated backward passes:
  `n_prompts * ceil(d_model / dim_batch)`
- CPU lens matrix storage MiB:
  `n_source_layers * d_model * d_model * 4 / 2**20`
- approximate checkpoint storage MiB:
  same formula plus metadata note
- warning if `dim_batch > 1` on Evo2 before profiling
- warning if `model_kind=evo2_7b` and GPU VRAM < 48 GB
- warning if trying Evo2 20B/40B/1B in v0

Expected example outputs:
- Othello all source layers:
  - d_model = 512
  - n_source_layers = 7
  - matrix storage ≈ 7 MiB
  - n_prompts = 10
  - dim_batch = 128
  - backward passes = 40
- Evo2 7B selected layers:
  - d_model = 4096
  - n_source_layers = 4
  - matrix storage ≈ 256 MiB
  - n_prompts = 10
  - dim_batch = 1
  - backward passes = 40,960

================================================================================
TEST SPEC
================================================================================

1. `tests/test_transformer_lens_adapter.py`

Use a tiny HookedTransformer config if TransformerLens is installed.

Test:
- Construct tiny HookedTransformer or use a fixture.
- Wrap with `TransformerLensLensModel`.
- Fit on 2–4 integer prompts.
- Assert:
  - `n_layers` matches model config
  - `d_model` matches model config
  - encode returns `[1, seq]`
  - apply output has expected vocab dimension
  - lens shapes are correct

Skip if TransformerLens unavailable.

--------------------------------------------------------------------------------

2. `tests/test_evo2_adapter_tiny.py`

Do not require real Evo2 weights.

Build a tiny fake StripedHyena-like module:
- `.blocks`: ModuleList of residual blocks
- each residual block should return either hidden or `(hidden, aux)` to test tuple handling
- `.norm`
- `.unembed`
- `.config.num_layers`
- `.config.hidden_size`
- `.block_idx_to_device`
- fake tokenizer

Wrap with `Evo2LensModel` or lower-level constructor path.

Fit on 2 toy DNA strings.

Assert:
- encode works
- forward works with autograd
- unembed returns expected vocab dimension
- `ActivationRecorder` correctly handles tuple block outputs if needed
- lens application shape is correct

--------------------------------------------------------------------------------

3. `tests/test_othello_smoke.py`

Mark:
- slow
- integration
- requires internet

Behavior:
- Load actual OthelloGPT checkpoint.
- Fit 5 prompts only.
- Apply lens to one sample sequence.
- Assert top-k outputs are valid token IDs in `[0, 60]`.

Do not make quality assertions brittle.

================================================================================
IMPLEMENTATION DETAILS AND GOTCHAS
================================================================================

Autograd:
- Do not use `torch.no_grad()` in adapter forward paths used for fitting.
- Freeze model parameters, but preserve activation gradients.
- Set models to eval mode.

Activation recorder:
- Anthropic’s recorder may assume each layer output is a tensor.
- Some models may return tuples.
- If required, minimally adjust recorder/hook logic to capture the first tensor from tuple outputs.
- Keep this change generic and covered by tests.

Device handling:
- Othello should be simple single-device.
- Evo2 may have internal device maps.
- For Evo2:
  - send tokens to first block device;
  - move residuals to norm/unembed device in `unembed`;
  - avoid accidental CPU/GPU mixing.

Memory:
- J matrices are not the main VRAM cost.
- Matrix storage is CPU/checkpoint-heavy but manageable:
  - Othello: 512 x 512 x 4 bytes ≈ 1 MiB per layer
  - Evo2 7B: 4096 x 4096 x 4 bytes ≈ 64 MiB per layer
- VRAM is dominated by retained autograd graph and backward passes.
- For Evo2 7B, `dim_batch=1` is safest but slow.
- Increase `dim_batch` only after profiling.

Backward passes:
- Approximate count per prompt:
  `ceil(d_model / dim_batch)`
- Othello:
  - d_model=512, dim_batch=128 -> 4 backward passes per prompt
- Evo2 7B:
  - d_model=4096, dim_batch=1 -> 4096 backward passes per prompt
  - d_model=4096, dim_batch=8 -> 512 backward passes per prompt, but more VRAM

Othello dataset:
- Prefer simple generated legal Othello games if there is existing code nearby.
- If legal game generator is unavailable, use available OthelloGPT demo data or synthetic token sequences for plumbing.
- For the coffee demo, actual model-compatible move sequences are needed for meaningful next-move metrics.
- Legal-move metrics are optional in Stage 1.

Othello interpretation:
- Direct J-lens decodes move tokens.
- It does not directly decode:
  - black/white/blank square occupancy
  - board state
  - legal move set as a concept
- Legal move enrichment can be reported as suggestive, not conclusive.

Evo2 interpretation:
- Direct J-lens decodes Evo2 output vocabulary tokens.
- It does not directly decode:
  - promoters
  - motifs
  - splice sites
  - enhancer concepts
  - biological mechanisms
- Such concepts require template/probe/concept extensions later.

Stage guards:
- `othello-coffee` must not import Evo2.
- `othello-coffee` must complete even if Evo2 is not installed.
- Any `evo2-*` real fitting command must require `--allow-real-fit`.
- Any Evo2 output must include token-level caveat.
- Any Othello output must include move-token caveat.

================================================================================
DOCUMENTATION SPEC
================================================================================

Add `docs/othello_evo2_jlens.md`.

Include sections:

1. What this implements
- Minimal adapter-based use of Anthropic Jacobian Lens.
- Reuses core J-lens fitting/apply/merge code.
- Adds model wrappers and staged scripts.

2. Why OthelloGPT first
- Tiny non-language model.
- Fast enough for same-day demo.
- Tests whether J-lens can expose useful output-token readouts outside natural language.
- Avoids Evo2 hardware/debug complexity.

3. Coffee demo command
Include:

  python scripts/run_othello_coffee_demo.py \
    --device cuda \
    --n-prompts 10 \
    --dim-batch 128 \
    --out-dir out/coffee

And fallback:

  python scripts/run_othello_coffee_demo.py \
    --device cuda \
    --n-prompts 3 \
    --source-layers 5,6 \
    --target-layer 7 \
    --dim-batch 32 \
    --out-dir out/coffee_fast

4. How to interpret Othello results
- Move-token readout.
- Compare to logit lens.
- Do not claim board-state decoding.
- Board-state claims require later probe/template extension.

5. Evo2 staging
Include table:

  Stage 0: fake tests/adapters — CPU
  Stage 1: Othello coffee demo — 8–16 GB
  Stage 2: Othello 100-prompt run — 16–24 GB
  Stage 3: Evo2 adapter plumbing — CPU/24 GB optional
  Stage 4: Evo2 7B one-layer tiny fit — 48 GB preferred, 24 GB risky
  Stage 5: Evo2 7B first real single-layer fit — 80 GB
  Stage 6: Evo2 7B selected layers — 80 GB
  Stage 7: Evo2 shards and merge — multiple/repeated 80 GB jobs

6. Evo2 commands
Include Stage 4, 5, and 6 commands from this spec.

7. Limitations
- No sparse nonnegative J-space decomposition in v0.
- No template lens in v0.
- No board-state probe bridge in v0.
- No biological concept readout in v0.
- Evo2 backward may fail on optimized kernels or Vortex internals.
- Evo2 20B/40B are out of scope.
- J-lens quality depends on prompt distribution.

================================================================================
FINAL DEFINITION OF DONE
================================================================================

Minimum done for today’s coffee demo:
1. `python scripts/run_othello_coffee_demo.py --device cuda --n-prompts 10 --dim-batch 128 --out-dir out/coffee` runs.
2. `out/coffee/othello_jlens.pt` exists.
3. `out/coffee/othello_eval.json` exists.
4. `out/coffee/othello_summary.md` exists.
5. The markdown includes per-layer next-move pass@5 and median rank.
6. The markdown includes logit-lens baseline.
7. The markdown includes 3 example prefixes with top-5 move-token readouts.
8. The markdown includes the required caveat:
   “This direct J-lens decodes move tokens, not board-state labels. Legal-move enrichment is suggestive only; board occupancy requires a probe/template extension.”

Minimum done for v0 branch:
1. Existing `jlens` tests still pass.
2. TransformerLens/Othello adapter works.
3. Othello coffee demo works.
4. Othello better eval works or is documented as optional.
5. Evo2 adapter fake tests pass.
6. Evo2 real fitting is available behind `--allow-real-fit`.
7. Evo2 apply script can run against a saved or tiny test lens.
8. Cost estimator works.
9. Docs explain GPU stages and caveats.
10. All outputs distinguish direct token-level J-lens readout from stronger semantic claims.

Implementation priority right now:
Start with Stage 1. Do not spend time on Evo2 until the Othello coffee demo produces `out/coffee/othello_summary.md`.

================================================================================
CODE MANAGEMENT STRATEGY FOR EPHEMERAL GPU AGENT WORK
================================================================================

Operational model:
The GPU machine is disposable. GitHub is the source of truth.

Hard rule:
Never allow the only copy of working code, instructions, logs, or results to live only on the RunPod / cloud GPU instance.

Canonical workflow:
- Work on a GitHub feature branch.
- Commit small coherent milestones.
- Push after each milestone.
- Keep generated model weights, caches, and large tensors out of git.
- Commit lightweight human-readable outputs such as markdown summaries, JSON metrics, and logs when useful.
- Before stopping the pod, confirm all code is pushed and artifacts are copied or packaged.

Recommended branch:
  jlens-othello-coffee

Recommended initial setup on the pod:
  git clone git@github.com:<OWNER>/<REPO>.git
  cd <REPO>
  git checkout -b jlens-othello-coffee origin/main

If SSH is unavailable, use HTTPS plus GitHub CLI/device auth or a short-lived fine-grained token. Do not copy a normal long-lived private SSH key into the pod.

Credentials policy:
- Preferred: GitHub CLI auth or a short-lived fine-grained GitHub token.
- Acceptable: repo-scoped deploy key for this repository only.
- Avoid: copying a personal main SSH private key onto the pod.
- Never commit tokens, keys, `.env` files, Hugging Face tokens, RunPod credentials, or API keys.
- If a credential appears in logs, do not commit those logs.

Create or update `AGENTS.md` at repo root with the current task:

  cat > AGENTS.md <<'EOF'
  Current task: Stage 1 OthelloGPT Jacobian Lens coffee demo.

  Hard constraints:
  - Do not work on Evo2 yet.
  - Do not import Evo2 in Othello scripts.
  - Reuse Anthropic jacobian-lens core logic.
  - Add adapters/scripts/tests only.
  - Produce out/coffee/othello_summary.md.
  - Commit and push after each coherent milestone.

  Run target:
  python scripts/run_othello_coffee_demo.py \
    --device cuda \
    --n-prompts 10 \
    --dim-batch 128 \
    --out-dir out/coffee

  Fallback:
  python scripts/run_othello_coffee_demo.py \
    --device cuda \
    --n-prompts 3 \
    --source-layers 5,6 \
    --target-layer 7 \
    --dim-batch 32 \
    --out-dir out/coffee_fast

  Stage boundary:
  Stop after Stage 1 succeeds. Do not start Evo2 unless explicitly instructed.
  EOF

Commit this before substantive implementation work:
  git add AGENTS.md
  git commit -m "Add agent instructions for Othello J-lens demo"
  git push -u origin jlens-othello-coffee

Repository hygiene:
Use this layout:
  repo/
    jlens/                 # code
    scripts/               # runnable entrypoints
    tests/                 # tests
    docs/                  # durable docs
    out/                   # generated outputs; mostly ignored
    data/                  # small sample data only
    .cache/                # model/HF caches; ignored
    logs/                  # runtime logs; mostly ignored except selected demo logs

Update `.gitignore`:

  out/**/*.pt
  out/**/*.pth
  out/**/*.bin
  out/**/*.safetensors
  .cache/
  __pycache__/
  *.pyc
  wandb/
  runs/
  .env
  *.log

  !out/coffee/
  !out/coffee/othello_summary.md
  !out/coffee/othello_eval.json
  !logs/
  !logs/othello_coffee.log

Do not commit:
- model checkpoints from Hugging Face;
- Evo2 weights;
- large `.pt`, `.pth`, `.bin`, `.safetensors`;
- local Python environments;
- caches;
- secrets;
- raw heavyweight experiment dumps.

Usually commit:
- source code;
- tests;
- docs;
- `AGENTS.md`;
- small config files;
- `out/coffee/othello_summary.md`;
- `out/coffee/othello_eval.json`;
- selected runtime logs such as `logs/othello_coffee.log`.

Othello lens artifact policy:
The fitted Othello `.pt` lens may be small, but default to not committing it. Treat it as a generated artifact. Package it separately if needed.

Before starting work, the agent must run:
  git status
  git branch --show-current
  git remote -v

If not on the intended feature branch, switch or create it before editing:
  git checkout -b jlens-othello-coffee origin/main

Use `tmux` for GPU runs:
  tmux new -s jlens

Log the main run:
  mkdir -p logs
  python scripts/run_othello_coffee_demo.py \
    --device cuda \
    --n-prompts 10 \
    --dim-batch 128 \
    --out-dir out/coffee 2>&1 | tee logs/othello_coffee.log

Fallback run:
  mkdir -p logs
  python scripts/run_othello_coffee_demo.py \
    --device cuda \
    --n-prompts 3 \
    --source-layers 5,6 \
    --target-layer 7 \
    --dim-batch 32 \
    --out-dir out/coffee_fast 2>&1 | tee logs/othello_coffee_fast.log

Commit cadence:
Commit after each coherent milestone.

Milestone 1:
- `AGENTS.md`
- `.gitignore`
- any packaging/dependency notes

Commit:
  git add AGENTS.md .gitignore
  git commit -m "Set up Othello J-lens agent workflow"
  git push

Milestone 2:
- TransformerLens adapter compiles
- minimal tests pass

Commit:
  git add jlens/adapters tests
  git commit -m "Add TransformerLens adapter for Jacobian Lens"
  git push

Milestone 3:
- Othello fit script runs or reaches a clear model-loading point

Commit:
  git add scripts/fit_othello_jlens.py scripts/run_othello_coffee_demo.py
  git commit -m "Add Othello Jacobian Lens fitting script"
  git push

Milestone 4:
- Othello eval script produces JSON/markdown

Commit:
  git add scripts/eval_othello_jlens.py docs tests
  git commit -m "Add Othello Jacobian Lens evaluation"
  git push

Milestone 5:
- Coffee demo outputs exist

Commit:
  git add out/coffee/othello_summary.md out/coffee/othello_eval.json logs/othello_coffee.log
  git commit -m "Add Othello J-lens coffee demo results"
  git push

If `git push` fails:
- Do not proceed as if work is durable.
- Keep local commits.
- Print exact recovery commands.
- Create a patch and artifact tarball:

  git format-patch origin/main --stdout > jlens_othello_coffee.patch
  tar -czf othello_coffee_artifacts.tgz \
    AGENTS.md \
    jlens \
    scripts \
    tests \
    docs \
    out/coffee \
    logs \
    jlens_othello_coffee.patch

- The user can download `othello_coffee_artifacts.tgz` before pod shutdown.

Pre-shutdown checklist:
Run this before stopping the pod:

  git status
  git log --oneline -8
  git push
  ls -lh out/coffee || true
  ls -lh logs || true
  tar -czf othello_coffee_artifacts.tgz out/coffee logs || true

Expected clean end state:
- Branch `jlens-othello-coffee` exists on GitHub.
- Code changes are committed and pushed.
- `out/coffee/othello_summary.md` is committed.
- `out/coffee/othello_eval.json` is committed.
- Main run log is committed or packaged.
- Large generated tensors are ignored or packaged separately.
- No secrets are committed.
- The pod can be deleted without losing the project state.

Agent stopping rule:
Once Stage 1 produces `out/coffee/othello_summary.md`, stop. Do not begin Evo2 implementation or fitting unless explicitly instructed.

================================================================================
RUNPOD / CLOUD GPU EXECUTION EXPECTATIONS
================================================================================

RunPod is an execution environment only. Do not treat the pod disk as durable.

Recommended first GPU:
- RTX 4090 if available.
- Otherwise RTX 3090, L4, A5000, or similar 16–24 GB card.
- Do not use A100/H100 for the Othello coffee demo unless cheaper GPUs are unavailable.

Storage:
- 30–60 GB container disk is enough for Stage 1.
- Persistent volume is optional but useful.
- Hugging Face cache can live on the pod; do not commit it.

Environment:
- Use a PyTorch/CUDA-ready image.
- Install only dependencies needed for Stage 1.
- Do not install Evo2 for the coffee demo unless already present.
- Othello scripts must run even if Evo2 is not installed.

Suggested setup commands:
  python --version
  nvidia-smi
  pip install -U pip
  pip install -e .
  pip install transformer-lens huggingface_hub pytest tqdm

If Anthropic’s `jacobian-lens` is not already vendored:
- Prefer adding it as a dependency or submodule clearly.
- If vendoring, keep changes minimal and documented.
- Do not rewrite the algorithm.

Run target:
  python scripts/run_othello_coffee_demo.py \
    --device cuda \
    --n-prompts 10 \
    --dim-batch 128 \
    --out-dir out/coffee

Immediate fallback:
  python scripts/run_othello_coffee_demo.py \
    --device cuda \
    --n-prompts 3 \
    --source-layers 5,6 \
    --target-layer 7 \
    --dim-batch 32 \
    --out-dir out/coffee_fast

CPU emergency fallback:
  python scripts/run_othello_coffee_demo.py \
    --device cpu \
    --n-prompts 1 \
    --source-layers 6 \
    --target-layer 7 \
    --dim-batch 8 \
    --out-dir out/coffee_cpu

The coffee demo is successful if:
- the command completes;
- `out/coffee/othello_summary.md` exists;
- `out/coffee/othello_eval.json` exists;
- outputs distinguish J-lens move-token readout from board-state decoding;
- code and lightweight outputs are pushed to GitHub.

================================================================================
UPDATE TO FINAL DEFINITION OF DONE
================================================================================

Minimum done for today’s coffee demo:
1. Work occurs on branch `jlens-othello-coffee`.
2. `AGENTS.md` exists and records the Stage 1-only instructions.
3. `.gitignore` protects checkpoints, caches, logs by default, and secrets.
4. TransformerLens/Othello code is committed and pushed.
5. `python scripts/run_othello_coffee_demo.py --device cuda --n-prompts 10 --dim-batch 128 --out-dir out/coffee` runs, or the documented fallback runs.
6. `out/coffee/othello_summary.md` exists.
7. `out/coffee/othello_eval.json` exists.
8. The markdown includes per-layer next-move pass@5 and median rank.
9. The markdown includes logit-lens baseline.
10. The markdown includes 3 example prefixes with top-5 move-token readouts.
11. The markdown includes the required caveat:
    “This direct J-lens decodes move tokens, not board-state labels. Legal-move enrichment is suggestive only; board occupancy requires a probe/template extension.”
12. `out/coffee/othello_summary.md`, `out/coffee/othello_eval.json`, and the relevant run log are committed or packaged.
13. `git push` has succeeded, or `jlens_othello_coffee.patch` and `othello_coffee_artifacts.tgz` have been created for download.
14. No Evo2 work has started.

Minimum done before stopping the pod:
1. `git status` has been checked.
2. All commits have been pushed, or patch/artifact tarball exists.
3. `out/coffee` has been listed and packaged.
4. No large model weights or secrets are staged.
5. The pod can be deleted without losing code or coffee-demo outputs.
