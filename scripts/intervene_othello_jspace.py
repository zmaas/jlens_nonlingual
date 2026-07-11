#!/usr/bin/env python3
"""Causally steer OthelloGPT along move-token J-lens vectors."""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from othello_common import (
    TOKEN_ENCODING,
    TOKEN_TO_SQUARE,
    game_position_states,
    generate_games,
    load_model,
    token_label,
    write_json,
)
from precoffee_common import (
    line_chart_svg,
    modify_block_output,
    summarize_grouped,
    validate_lens_metadata,
)

METHODS = ("jlens", "logit", "random", "other_jlens")
METRICS = (
    "delta_target_logit",
    "rank_improvement",
    "target_top5",
    "delta_legal_probability_mass",
    "output_kl",
)


def _rank(logits, target: int) -> int:
    return int((logits > logits[target]).sum().item()) + 1


def _unit(vector):
    return vector / vector.norm().clamp_min(1e-12)


def _j_direction(lens, unembed, layer: int, token: int, device):
    jacobian = lens.jacobians[layer].to(device=device, dtype=unembed.dtype)
    return _unit(jacobian.T @ unembed[:, token])


def _choose_targets(
    record: dict, baseline_logits, *, include_illegal: bool
) -> list[tuple[str, int]]:
    import torch

    legal = set(record["legal_tokens"])
    ranking = torch.argsort(baseline_logits, descending=True).tolist()
    outside_top5 = [token for token in ranking[5:] if token in legal]
    legal_target = (
        outside_top5[0] if outside_top5 else next(token for token in ranking if token in legal)
    )
    targets = [("legal", legal_target)]
    if include_illegal:
        empty_illegal = [
            token
            for token in ranking
            if token != 0
            and token not in legal
            and record["board"][TOKEN_TO_SQUARE[token - 1]] == 0
        ]
        illegal = (
            empty_illegal[0]
            if empty_illegal
            else next(token for token in ranking if token != 0 and token not in legal)
        )
        targets.append(("illegal", illegal))
    return targets


def _baseline(adapter, input_ids, layers: list[int], position: int):
    from jlens import ActivationRecorder

    with ActivationRecorder(adapter.layers, at=[*layers, adapter.n_layers - 1]) as recorder:
        adapter.forward(input_ids)
    activations = {
        layer: recorder.activations[layer][0, position].detach().float() for layer in layers
    }
    final_residual = recorder.activations[adapter.n_layers - 1][0, position].detach()
    logits = adapter.unembed(final_residual.float()).float().cpu()
    return activations, logits


def _steered_logits(adapter, input_ids, *, layer: int, direction, delta_norm: float, position: int):
    from jlens import ActivationRecorder

    def transform(tensor):
        changed = tensor.clone()
        changed[:, position, :] += direction.to(tensor) * delta_norm
        return changed

    handle = adapter.layers[layer].register_forward_hook(
        lambda _module, _inputs, output: modify_block_output(output, transform)
    )
    try:
        with ActivationRecorder(adapter.layers, at=[adapter.n_layers - 1]) as recorder:
            adapter.forward(input_ids)
        final = recorder.activations[adapter.n_layers - 1][0, position].detach()
        return adapter.unembed(final.float()).float().cpu()
    finally:
        handle.remove()


def run(args) -> dict:
    import torch
    from jlens import JacobianLens
    from jlens.adapters import TransformerLensLensModel

    metadata = validate_lens_metadata(args.lens)
    lens = JacobianLens.load(args.lens)
    adapter = TransformerLensLensModel(load_model(args.device, args.checkpoint))
    games = generate_games(args.n_boards, seed=args.seed, min_length=args.skip_first + 3)
    alphas = [float(value) for value in args.alphas.split(",")]
    unembed = adapter.model.W_U.detach().to(adapter.input_device).float()
    rng = random.Random(args.control_seed)
    rows = []

    for board_index, game in enumerate(games):
        records = game_position_states(game, skip_first=args.skip_first)
        row_index = min(len(records) - 1, int(args.position_fraction * len(records)))
        record = records[row_index]
        position = record["position"]
        input_ids = adapter.encode(game, max_length=59)
        activations, baseline_logits = _baseline(adapter, input_ids, lens.source_layers, position)
        baseline_probabilities = torch.softmax(baseline_logits, dim=-1)
        baseline_legal_mass = float(baseline_probabilities[record["legal_tokens"]].sum())
        targets = _choose_targets(record, baseline_logits, include_illegal=args.include_illegal)
        for condition, target in targets:
            other_candidates = [token for token in record["legal_tokens"] if token != target]
            other_token = other_candidates[0] if other_candidates else (1 if target != 1 else 2)
            baseline_rank = _rank(baseline_logits, target)
            for layer in lens.source_layers:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(rng.randrange(2**31))
                random_direction = _unit(torch.randn(adapter.d_model, generator=generator)).to(
                    adapter.input_device
                )
                directions = {
                    "jlens": _j_direction(lens, unembed, layer, target, adapter.input_device),
                    "logit": _unit(unembed[:, target]),
                    "random": random_direction,
                    "other_jlens": _j_direction(
                        lens, unembed, layer, other_token, adapter.input_device
                    ),
                }
                activation_norm = float(activations[layer].norm())
                for alpha in alphas:
                    delta_norm = alpha * activation_norm
                    for method in METHODS:
                        logits = _steered_logits(
                            adapter,
                            input_ids,
                            layer=layer,
                            direction=directions[method],
                            delta_norm=delta_norm,
                            position=position,
                        )
                        probabilities = torch.softmax(logits, dim=-1)
                        legal_mass = float(probabilities[record["legal_tokens"]].sum())
                        output_kl = float(
                            (
                                baseline_probabilities
                                * (
                                    baseline_probabilities.clamp_min(1e-12).log()
                                    - probabilities.clamp_min(1e-12).log()
                                )
                            ).sum()
                        )
                        rank = _rank(logits, target)
                        rows.append(
                            {
                                "board_index": board_index,
                                "position": position,
                                "condition": condition,
                                "target": target,
                                "target_label": token_label(target),
                                "other_token": other_token,
                                "layer": layer,
                                "alpha": alpha,
                                "method": method,
                                "delta_target_logit": float(
                                    logits[target] - baseline_logits[target]
                                ),
                                "rank_improvement": baseline_rank - rank,
                                "target_top5": float(rank <= 5),
                                "delta_legal_probability_mass": (legal_mass - baseline_legal_mass),
                                "output_kl": output_kl,
                            }
                        )

    by_key_board: dict[tuple, dict[int, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: {metric: [] for metric in METRICS})
    )
    for row in rows:
        key = (row["condition"], row["method"], row["layer"], row["alpha"])
        for metric in METRICS:
            by_key_board[key][row["board_index"]][metric].append(float(row[metric]))
    summary = {}
    for key, boards in by_key_board.items():
        condition, method, layer, alpha = key
        key_name = f"{condition}|{method}|L{layer}|a{alpha:g}"
        summary[key_name] = {}
        for metric in METRICS:
            per_board = [values[metric] for values in boards.values()]
            summary[key_name][metric] = summarize_grouped(
                per_board,
                n_bootstrap=args.n_bootstrap,
                seed=args.bootstrap_seed,
            )
    return {
        "experiment": "othello_causal_jspace_steering",
        "token_encoding": TOKEN_ENCODING,
        "lens_metadata": metadata,
        "n_boards": args.n_boards,
        "alphas": alphas,
        "position_fraction": args.position_fraction,
        "methods": list(METHODS),
        "summary": summary,
        "rows": rows,
    }


def write_artifacts(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "steering_results.json", result)
    layers = result["lens_metadata"]["source_layers"]
    alpha = max(result["alphas"])
    series = {}
    for method in METHODS:
        series[method] = [
            result["summary"][f"legal|{method}|L{layer}|a{alpha:g}"]["delta_target_logit"]["mean"]
            for layer in layers
        ]
    line_chart_svg(
        title=f"Legal target-logit effect (alpha={alpha:g} residual norm)",
        y_label="Delta target logit",
        x_values=layers,
        series=series,
        path=out_dir / "target_logit_effect_by_layer.svg",
    )
    lines = [
        "# Othello causal J-space steering",
        "",
        f"Boards: {result['n_boards']}; intervention norm fractions: {result['alphas']}.",
        "",
        f"## Strongest intervention (alpha={alpha:g})",
        "",
        "| Layer | J delta logit | Logit-direction delta | Random delta | Other-J delta |",
        "|---:|---:|---:|---:|---:|",
    ]
    for index, layer in enumerate(layers):
        lines.append(
            f"| {layer} | {series['jlens'][index]:.3f} | "
            f"{series['logit'][index]:.3f} | {series['random'][index]:.3f} | "
            f"{series['other_jlens'][index]:.3f} |"
        )
    lines += [
        "",
        "![Target-logit effect](target_logit_effect_by_layer.svg)",
        "",
        "Positive, target-specific effects beyond random and alternate-move controls "
        "support a reusable, causally writable move representation.",
        "",
    ]
    (out_dir / "steering_summary.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint")
    parser.add_argument("--lens", default="out/coffee_v2/othello_jlens.pt")
    parser.add_argument("--n-boards", type=int, default=40)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--skip-first", type=int, default=16)
    parser.add_argument("--position-fraction", type=float, default=0.6)
    parser.add_argument("--alphas", default="0.02,0.05,0.1")
    parser.add_argument("--include-illegal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--control-seed", type=int, default=0)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--out-dir", default="out/precoffee/steering")
    args = parser.parse_args()
    result = run(args)
    write_artifacts(result, Path(args.out_dir))
    print(f"saved steering artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()
