#!/usr/bin/env python3
"""Measure legal-move enrichment for J-lens, logit lens, and final logits."""

from __future__ import annotations

import argparse
from pathlib import Path

from othello_common import (
    CAVEAT,
    TOKEN_ENCODING,
    game_position_states,
    generate_games,
    load_model,
    write_json,
)
from precoffee_common import (
    line_chart_svg,
    summarize_grouped,
    validate_lens_metadata,
)

METRICS = (
    "legal_precision_at_k",
    "legal_recall_at_k",
    "legal_probability_mass",
    "any_legal_at_k",
    "unused_token_at_k",
    "target_pass_at_k",
)


def _position_metrics(logits, legal_tokens: list[int], target: int, k: int) -> dict[str, float]:
    import torch

    top = torch.topk(logits, k).indices.tolist()
    legal = set(legal_tokens)
    hits = sum(token in legal for token in top)
    probabilities = torch.softmax(logits.float(), dim=-1)
    return {
        "legal_precision_at_k": hits / k,
        "legal_recall_at_k": hits / len(legal),
        "legal_probability_mass": float(probabilities[legal_tokens].sum()),
        "any_legal_at_k": float(hits > 0),
        "unused_token_at_k": float(0 in top),
        "target_pass_at_k": float(target in top),
    }


def evaluate(args) -> dict:
    from jlens import JacobianLens
    from jlens.adapters import TransformerLensLensModel

    metadata = validate_lens_metadata(args.lens)
    model = TransformerLensLensModel(load_model(args.device, args.checkpoint))
    lens = JacobianLens.load(args.lens)
    games = generate_games(args.n_games, seed=args.seed, min_length=args.skip_first + 2)
    method_names = [
        *(f"jlens_L{layer}" for layer in lens.source_layers),
        *(f"logit_L{layer}" for layer in lens.source_layers),
        "final",
    ]
    grouped = {method: {metric: [] for metric in METRICS} for method in method_names}

    for game in games:
        records = game_position_states(game, skip_first=args.skip_first)
        positions = [record["position"] for record in records]
        j_logits, final_logits, _ = lens.apply(model, game, positions=positions, max_seq_len=59)
        logit_logits, _, _ = lens.apply(
            model,
            game,
            positions=positions,
            max_seq_len=59,
            use_jacobian=False,
        )
        per_game = {method: {metric: [] for metric in METRICS} for method in method_names}
        for row, record in enumerate(records):
            candidates = {"final": final_logits[row]}
            for layer in lens.source_layers:
                candidates[f"jlens_L{layer}"] = j_logits[layer][row]
                candidates[f"logit_L{layer}"] = logit_logits[layer][row]
            for method, logits in candidates.items():
                values = _position_metrics(logits, record["legal_tokens"], record["target"], args.k)
                for metric, value in values.items():
                    per_game[method][metric].append(value)
        for method in method_names:
            for metric in METRICS:
                grouped[method][metric].append(per_game[method][metric])

    metrics = {
        method: {
            metric: summarize_grouped(
                grouped[method][metric],
                n_bootstrap=args.n_bootstrap,
                seed=args.bootstrap_seed,
            )
            for metric in METRICS
        }
        for method in method_names
    }
    paired = {}
    for layer in lens.source_layers:
        paired[str(layer)] = {}
        for metric in METRICS:
            differences = [
                [j - baseline for j, baseline in zip(j_game, b_game, strict=True)]
                for j_game, b_game in zip(
                    grouped[f"jlens_L{layer}"][metric],
                    grouped[f"logit_L{layer}"][metric],
                    strict=True,
                )
            ]
            paired[str(layer)][metric] = summarize_grouped(
                differences,
                n_bootstrap=args.n_bootstrap,
                seed=args.bootstrap_seed,
            )

    return {
        "experiment": "othello_legal_move_emergence",
        "token_encoding": TOKEN_ENCODING,
        "lens_metadata": metadata,
        "n_eval_games": args.n_games,
        "k": args.k,
        "source_layers": lens.source_layers,
        "metrics": metrics,
        "paired_jlens_minus_logit_lens": paired,
        "caveat": CAVEAT,
    }


def write_artifacts(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "legal_eval.json", result)
    layers = result["source_layers"]
    final_precision = result["metrics"]["final"]["legal_precision_at_k"]["mean"]
    final_mass = result["metrics"]["final"]["legal_probability_mass"]["mean"]
    line_chart_svg(
        title="Legal precision@5 by layer",
        y_label="Precision",
        x_values=layers,
        series={
            "J-lens": [
                result["metrics"][f"jlens_L{layer}"]["legal_precision_at_k"]["mean"]
                for layer in layers
            ],
            "Logit lens": [
                result["metrics"][f"logit_L{layer}"]["legal_precision_at_k"]["mean"]
                for layer in layers
            ],
            "Final": [final_precision] * len(layers),
        },
        path=out_dir / "legal_precision_by_layer.svg",
        y_min=0,
        y_max=1,
        percent=True,
    )
    line_chart_svg(
        title="Probability mass on legal moves",
        y_label="Legal probability mass",
        x_values=layers,
        series={
            "J-lens": [
                result["metrics"][f"jlens_L{layer}"]["legal_probability_mass"]["mean"]
                for layer in layers
            ],
            "Logit lens": [
                result["metrics"][f"logit_L{layer}"]["legal_probability_mass"]["mean"]
                for layer in layers
            ],
            "Final": [final_mass] * len(layers),
        },
        path=out_dir / "legal_mass_by_layer.svg",
        y_min=0,
        y_max=1,
        percent=True,
    )
    lines = [
        "# Othello legal-move emergence",
        "",
        f"Games: {result['n_eval_games']}; top-k: {result['k']}.",
        "",
        "| Layer | J precision@5 | Logit precision@5 | J legal mass | Logit legal mass |",
        "|---:|---:|---:|---:|---:|",
    ]
    for layer in layers:
        j = result["metrics"][f"jlens_L{layer}"]
        logit = result["metrics"][f"logit_L{layer}"]
        lines.append(
            f"| {layer} | {j['legal_precision_at_k']['mean']:.3f} | "
            f"{logit['legal_precision_at_k']['mean']:.3f} | "
            f"{j['legal_probability_mass']['mean']:.3f} | "
            f"{logit['legal_probability_mass']['mean']:.3f} |"
        )
    lines += [
        f"| final | {final_precision:.3f} | — | {final_mass:.3f} | — |",
        "",
        "![Legal precision](legal_precision_by_layer.svg)",
        "",
        "![Legal probability mass](legal_mass_by_layer.svg)",
        "",
        result["caveat"],
        "",
    ]
    (out_dir / "legal_summary.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint")
    parser.add_argument("--lens", default="out/coffee_v2/othello_jlens.pt")
    parser.add_argument("--n-games", type=int, default=100)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--skip-first", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--out-dir", default="out/precoffee/legal")
    args = parser.parse_args()
    result = evaluate(args)
    write_artifacts(result, Path(args.out_dir))
    print(f"saved legality artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()
