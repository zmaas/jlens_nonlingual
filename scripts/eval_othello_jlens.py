#!/usr/bin/env python3
"""Evaluate J-lens, logit-lens, and final logits on held-out Othello games."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from othello_common import (
    CAVEAT,
    TOKEN_ENCODING,
    generate_games,
    load_model,
    token_label,
    write_json,
)


def _rank(logits, target: int) -> int:
    return int((logits > logits[target]).sum().item()) + 1


def _summary(ranks: list[int], k: int) -> dict[str, float | int]:
    return {
        f"pass@{k}": sum(rank <= k for rank in ranks) / len(ranks),
        "median_rank": float(statistics.median(ranks)),
        "n_predictions": len(ranks),
    }


def evaluate(args) -> dict:
    import torch
    from jlens import JacobianLens
    from jlens.adapters import TransformerLensLensModel

    metadata_path = Path(args.lens).with_suffix(".metadata.json")
    if not metadata_path.exists():
        raise RuntimeError(f"missing lens metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("token_encoding") != TOKEN_ENCODING:
        raise RuntimeError(
            f"refusing to evaluate {args.lens}: expected token encoding "
            f"{TOKEN_ENCODING!r}, found {metadata.get('token_encoding')!r}; refit the lens"
        )
    model = TransformerLensLensModel(load_model(args.device, args.checkpoint))
    lens = JacobianLens.load(args.lens)
    games = generate_games(args.n_games, seed=args.seed, min_length=args.skip_first + 2)
    ranks = {
        "jlens": {str(layer): [] for layer in lens.source_layers},
        "logit_lens": {str(layer): [] for layer in lens.source_layers},
        "final_logits": [],
    }
    examples = []
    for game_index, game in enumerate(games):
        positions = list(range(args.skip_first, len(game) - 1))
        j_logits, final_logits, _ = lens.apply(model, game, positions=positions, max_seq_len=59)
        l_logits, _, _ = lens.apply(
            model, game, positions=positions, max_seq_len=59, use_jacobian=False
        )
        targets = game[args.skip_first + 1 :]
        for row, target in enumerate(targets):
            ranks["final_logits"].append(_rank(final_logits[row], target))
            for layer in lens.source_layers:
                ranks["jlens"][str(layer)].append(_rank(j_logits[layer][row], target))
                ranks["logit_lens"][str(layer)].append(_rank(l_logits[layer][row], target))
        if game_index < 3:
            # Three shareable prefixes total: early-midgame, late-midgame, and
            # the last position that still has a next-move target.
            candidate_rows = [len(positions) // 3, 2 * len(positions) // 3, len(positions) - 1]
            row = candidate_rows[game_index]
            position = positions[row]
            per_layer = {}
            for layer in lens.source_layers:
                top = torch.topk(j_logits[layer][row], args.k).indices.tolist()
                per_layer[str(layer)] = [token_label(token) for token in top]
            examples.append(
                {
                    "game_index": game_index,
                    "position": position,
                    "prefix": [token_label(t) for t in game[: position + 1]],
                    "target": token_label(game[position + 1]),
                    "jlens_top_k_by_layer": per_layer,
                }
            )

    return {
        "model": "OthelloGPT synthetic TransformerLens checkpoint",
        "config": {"n_layers": 8, "d_model": 512, "d_vocab": 61, "n_ctx": 59},
        "source_layers": lens.source_layers,
        "target_layer": metadata.get("target_layer", 7),
        "n_prompts": lens.n_prompts,
        "n_eval_games": args.n_games,
        "k": args.k,
        "token_encoding": TOKEN_ENCODING,
        "metrics": {
            "jlens": {layer: _summary(values, args.k) for layer, values in ranks["jlens"].items()},
            "logit_lens": {
                layer: _summary(values, args.k) for layer, values in ranks["logit_lens"].items()
            },
            "final_logits": _summary(ranks["final_logits"], args.k),
        },
        "examples": examples,
        "caveat": CAVEAT,
    }


def markdown(result: dict) -> str:
    k = result["k"]
    lines = [
        "# OthelloGPT Jacobian Lens coffee demo",
        "",
        "## Configuration",
        "",
        f"- Model: {result['model']}",
        "- Config: 8 layers, d_model=512, vocabulary=61, context=59",
        f"- Source layers: {result['source_layers']}",
        f"- Target layer: {result['target_layer']}",
        f"- Fit prompts: {result['n_prompts']}",
        f"- Evaluation games: {result['n_eval_games']}",
        "",
        "## Next-move results",
        "",
        f"| Layer | J-lens pass@{k} | J-lens median rank | "
        f"Logit lens pass@{k} | Logit lens median rank |",
        "|---:|---:|---:|---:|---:|",
    ]
    for layer in result["source_layers"]:
        j = result["metrics"]["jlens"][str(layer)]
        logit = result["metrics"]["logit_lens"][str(layer)]
        lines.append(
            f"| {layer} | {j[f'pass@{k}']:.3f} | {j['median_rank']:.1f} | "
            f"{logit[f'pass@{k}']:.3f} | {logit['median_rank']:.1f} |"
        )
    final = result["metrics"]["final_logits"]
    lines += [
        f"| final logits | {final[f'pass@{k}']:.3f} | {final['median_rank']:.1f} | — | — |",
        "",
        "## Example prefixes",
        "",
    ]
    for example in result["examples"]:
        lines.append(f"### Game {example['game_index'] + 1}")
        lines.append("")
        prefix = " ".join(example["prefix"])
        lines += [
            f"Position {example['position']} — target `{example['target']}`",
            "",
            f"Prefix: `{prefix}`",
            "",
        ]
        for layer, tokens in example["jlens_top_k_by_layer"].items():
            lines.append(f"- L{layer}: " + ", ".join(f"`{t}`" for t in tokens))
        lines.append("")
    lines += [
        "## Interpretation",
        "",
        CAVEAT,
        "",
        "- We reused Anthropic’s average-Jacobian lens design unchanged.",
        "- The new work is an adapter: TransformerLens residual streams -> "
        "J-lens API -> Othello move-token unembedding.",
        "- This is the smallest non-language sanity check before Evo2.",
        "- The direct readout should be interpreted as “what move-token futures this "
        "activation linearly transports to,” not “the board state is decoded.”",
        "- Next step is a board-state template/probe lens if move-token readout is clean.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", help="Local synthetic_model.pth (avoids Hub download)")
    parser.add_argument("--lens", default="out/coffee/othello_jlens.pt")
    parser.add_argument("--n-games", type=int, default=25)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--skip-first", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default="out/coffee/othello_eval.json")
    parser.add_argument("--markdown", default="out/coffee/othello_summary.md")
    args = parser.parse_args()
    result = evaluate(args)
    write_json(args.out, result)
    path = Path(args.markdown)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown(result))
    print(f"saved {args.out} and {args.markdown}")


if __name__ == "__main__":
    main()
