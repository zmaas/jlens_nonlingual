#!/usr/bin/env python3
"""Probe and ablate OthelloGPT's J-space versus orthogonal computation."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from othello_common import (
    TOKEN_ENCODING,
    game_position_states,
    generate_games,
    load_model,
    write_json,
)
from precoffee_common import (
    line_chart_svg,
    modify_block_output,
    summarize_grouped,
    validate_lens_metadata,
)

COMPONENTS = ("full", "jspace", "orthogonal", "random_matched")
ABLATIONS = ("remove_jspace", "remove_logit", "remove_random", "keep_jspace")


def _orthonormal_basis(matrix, *, relative_tolerance: float = 1e-5):
    import torch

    u, singular_values, _ = torch.linalg.svd(matrix.float(), full_matrices=False)
    threshold = singular_values[0] * relative_tolerance
    rank = int((singular_values > threshold).sum())
    return u[:, :rank], singular_values


def _random_basis(d_model: int, rank: int, *, seed: int, device):
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    matrix = torch.randn(d_model, rank, generator=generator).to(device)
    return torch.linalg.qr(matrix, mode="reduced").Q


def _collect_probe_data(adapter, games, layers: list[int], *, skip_first: int):
    import torch
    from jlens import ActivationRecorder

    activations = {layer: [] for layer in layers}
    targets = []
    groups = []
    for game_index, game in enumerate(games):
        records = game_position_states(game, skip_first=skip_first)
        positions = [record["position"] for record in records]
        input_ids = adapter.encode(game, max_length=59)
        with ActivationRecorder(adapter.layers, at=layers) as recorder:
            adapter.forward(input_ids)
        for layer in layers:
            activations[layer].append(
                recorder.activations[layer][0, positions].detach().float().cpu()
            )
        for record in records:
            board = torch.tensor(record["board"], dtype=torch.float32)
            legal = torch.zeros(60)
            legal[[token - 1 for token in record["legal_tokens"]]] = 1
            player = torch.tensor([record["player"]], dtype=torch.float32)
            next_move = torch.zeros(60)
            next_move[record["target"] - 1] = 1
            targets.append(torch.cat([board, legal, player, next_move]))
            groups.append(game_index)
    return (
        {layer: torch.cat(values) for layer, values in activations.items()},
        torch.stack(targets),
        torch.tensor(groups),
    )


def _ridge_predict(train_x, test_x, train_y, *, ridge: float):
    import torch

    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-5)
    train = (train_x - mean) / std
    test = (test_x - mean) / std
    train = torch.cat([train, torch.ones(len(train), 1, device=train.device)], dim=1)
    test = torch.cat([test, torch.ones(len(test), 1, device=test.device)], dim=1)
    identity = torch.eye(train.shape[1], device=train.device)
    identity[-1, -1] = 0
    weights = torch.linalg.solve(
        train.T @ train + ridge * identity,
        train.T @ train_y,
    )
    return test @ weights


def _probe_metrics(predictions, targets, *, k: int = 5) -> dict[str, float]:
    import torch

    board_prediction = predictions[:, :64].round().clamp(-1, 1)
    board_accuracy = float((board_prediction == targets[:, :64]).float().mean())
    legal_scores = predictions[:, 64:124]
    legal_targets = targets[:, 64:124]
    legal_top = torch.topk(legal_scores, k, dim=-1).indices
    legal_hits = legal_targets.gather(1, legal_top).sum(dim=1)
    legal_precision = float((legal_hits / k).mean())
    legal_recall = float((legal_hits / legal_targets.sum(dim=1)).mean())
    player_accuracy = float(((predictions[:, 124] > 0) == (targets[:, 124] > 0)).float().mean())
    next_scores = predictions[:, 125:185]
    next_target = targets[:, 125:185].argmax(dim=-1)
    next_top = torch.topk(next_scores, k, dim=-1).indices
    next_pass = float((next_top == next_target[:, None]).any(dim=1).float().mean())
    return {
        "board_state_accuracy": board_accuracy,
        "legal_precision_at_5": legal_precision,
        "legal_recall_at_5": legal_recall,
        "player_accuracy": player_accuracy,
        "next_move_pass_at_5": next_pass,
    }


def _final_logits(adapter, input_ids, positions: list[int]):
    from jlens import ActivationRecorder

    final_layer = adapter.n_layers - 1
    with ActivationRecorder(adapter.layers, at=[final_layer]) as recorder:
        adapter.forward(input_ids)
    residual = recorder.activations[final_layer][0, positions].detach().float()
    return adapter.unembed(residual).float().cpu()


def _ablated_logits(adapter, input_ids, positions, *, layer: int, basis, mode: str):
    from jlens import ActivationRecorder

    def transform(tensor):
        q = basis.to(tensor)
        projection = (tensor @ q) @ q.T
        return projection if mode == "keep_jspace" else tensor - projection

    handle = adapter.layers[layer].register_forward_hook(
        lambda _module, _inputs, output: modify_block_output(output, transform)
    )
    try:
        final_layer = adapter.n_layers - 1
        with ActivationRecorder(adapter.layers, at=[final_layer]) as recorder:
            adapter.forward(input_ids)
        residual = recorder.activations[final_layer][0, positions].detach().float()
        return adapter.unembed(residual).float().cpu()
    finally:
        handle.remove()


def _behavior_metrics(logits, records, baseline_logits=None) -> dict[str, list[float]]:
    import torch

    values = defaultdict(list)
    for row, record in enumerate(records):
        legal = record["legal_tokens"]
        probabilities = torch.softmax(logits[row], dim=-1)
        top = torch.topk(logits[row], 5).indices.tolist()
        values["legal_precision_at_5"].append(sum(token in set(legal) for token in top) / 5)
        values["legal_probability_mass"].append(float(probabilities[legal].sum()))
        values["target_rank"].append(float((logits[row] > logits[row, record["target"]]).sum() + 1))
        if baseline_logits is not None:
            baseline_prob = torch.softmax(baseline_logits[row], dim=-1)
            kl = (
                baseline_prob
                * (baseline_prob.clamp_min(1e-12).log() - probabilities.clamp_min(1e-12).log())
            ).sum()
            values["output_kl"].append(float(kl))
    return dict(values)


def run(args) -> dict:
    import torch
    from jlens import JacobianLens
    from jlens.adapters import TransformerLensLensModel

    metadata = validate_lens_metadata(args.lens)
    lens = JacobianLens.load(args.lens)
    adapter = TransformerLensLensModel(load_model(args.device, args.checkpoint))
    layers = lens.source_layers
    unembed = adapter.model.W_U.detach().float().to(adapter.input_device)
    logit_basis, _ = _orthonormal_basis(unembed[:, 1:])
    bases = {}
    geometry = {}
    for layer in layers:
        jacobian = lens.jacobians[layer].to(adapter.input_device)
        directions = jacobian.T @ unembed[:, 1:]
        q, singular_values = _orthonormal_basis(directions)
        random_q = _random_basis(
            adapter.d_model, q.shape[1], seed=args.random_seed + layer, device=q.device
        )
        bases[layer] = {"jspace": q, "random": random_q, "logit": logit_basis}
        normalized = singular_values / singular_values.sum()
        entropy_terms = normalized * normalized.clamp_min(1e-12).log()
        effective_rank = float(torch.exp(-entropy_terms.sum()))
        geometry[str(layer)] = {
            "rank": q.shape[1],
            "effective_rank": effective_rank,
            "singular_values": singular_values.cpu().tolist(),
        }

    games = generate_games(args.n_probe_games, seed=args.probe_seed, min_length=args.skip_first + 2)
    activations, targets, groups = _collect_probe_data(
        adapter, games, layers, skip_first=args.skip_first
    )
    n_train_games = max(1, int(args.train_fraction * args.n_probe_games))
    train_mask = (groups < n_train_games).to(adapter.input_device)
    test_mask = ~train_mask
    probe_results = {}
    for layer in layers:
        x = activations[layer].to(adapter.input_device)
        y = targets.to(adapter.input_device)
        q = bases[layer]["jspace"]
        random_q = bases[layer]["random"]
        projected = (x @ q) @ q.T
        centered = x - x.mean(dim=0, keepdim=True)
        centered_projection = (centered @ q) @ q.T
        geometry[str(layer)]["activation_variance_fraction"] = float(
            centered_projection.square().sum() / centered.square().sum()
        )
        component_features = {
            "full": x,
            "jspace": x @ q,
            "orthogonal": x - projected,
            "random_matched": x @ random_q,
        }
        probe_results[str(layer)] = {}
        for component, features in component_features.items():
            predictions = _ridge_predict(
                features[train_mask],
                features[test_mask],
                y[train_mask],
                ridge=args.ridge,
            )
            probe_results[str(layer)][component] = _probe_metrics(predictions, y[test_mask])

    ablation_games = generate_games(
        args.n_ablation_games,
        seed=args.ablation_seed,
        min_length=args.skip_first + 2,
    )
    grouped = defaultdict(lambda: defaultdict(list))
    for game in ablation_games:
        records = game_position_states(game, skip_first=args.skip_first)
        positions = [record["position"] for record in records]
        input_ids = adapter.encode(game, max_length=59)
        baseline_logits = _final_logits(adapter, input_ids, positions)
        baseline_metrics = _behavior_metrics(baseline_logits, records)
        for metric, values in baseline_metrics.items():
            grouped[("baseline", -1)][metric].append(values)
        for layer in layers:
            basis_by_mode = {
                "remove_jspace": bases[layer]["jspace"],
                "remove_logit": bases[layer]["logit"],
                "remove_random": bases[layer]["random"],
                "keep_jspace": bases[layer]["jspace"],
            }
            for mode, basis in basis_by_mode.items():
                logits = _ablated_logits(
                    adapter,
                    input_ids,
                    positions,
                    layer=layer,
                    basis=basis,
                    mode=mode,
                )
                metrics = _behavior_metrics(logits, records, baseline_logits)
                for metric, values in metrics.items():
                    grouped[(mode, layer)][metric].append(values)

    ablations = {}
    for (mode, layer), metrics in grouped.items():
        key = mode if layer == -1 else f"{mode}|L{layer}"
        ablations[key] = {
            metric: summarize_grouped(
                per_game,
                n_bootstrap=args.n_bootstrap,
                seed=args.bootstrap_seed,
            )
            for metric, per_game in metrics.items()
        }
    return {
        "experiment": "othello_jspace_workspace_split",
        "projection_definition": (
            "Linear span of move-token J-lens directions. This is a tractable "
            "J-span proxy, not Anthropic's sparse nonnegative J-space cone."
        ),
        "token_encoding": TOKEN_ENCODING,
        "lens_metadata": metadata,
        "n_probe_games": args.n_probe_games,
        "n_ablation_games": args.n_ablation_games,
        "n_train_games": n_train_games,
        "n_test_games": args.n_probe_games - n_train_games,
        "geometry": geometry,
        "probes": probe_results,
        "ablations": ablations,
    }


def write_artifacts(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "workspace_split.json", result)
    layers = result["lens_metadata"]["source_layers"]
    line_chart_svg(
        title="Legal-move probe precision@5",
        y_label="Precision",
        x_values=layers,
        series={
            component: [
                result["probes"][str(layer)][component]["legal_precision_at_5"] for layer in layers
            ]
            for component in COMPONENTS
        },
        path=out_dir / "probe_accuracy_by_component.svg",
        y_min=0,
        y_max=1,
        percent=True,
    )
    line_chart_svg(
        title="Activation variance in J-space span",
        y_label="Variance fraction",
        x_values=layers,
        series={
            "J-space": [
                result["geometry"][str(layer)]["activation_variance_fraction"] for layer in layers
            ]
        },
        path=out_dir / "jspace_variance_by_layer.svg",
        y_min=0,
        percent=True,
    )
    baseline_mass = result["ablations"]["baseline"]["legal_probability_mass"]["mean"]
    line_chart_svg(
        title="Legal probability mass after subspace intervention",
        y_label="Legal probability mass",
        x_values=layers,
        series={
            mode: [
                result["ablations"][f"{mode}|L{layer}"]["legal_probability_mass"]["mean"]
                for layer in layers
            ]
            for mode in ABLATIONS
        }
        | {"baseline": [baseline_mass] * len(layers)},
        path=out_dir / "ablation_effect_by_layer.svg",
        y_min=0,
        y_max=1,
        percent=True,
    )
    lines = [
        "# Othello J-space versus orthogonal computation",
        "",
        result["projection_definition"],
        "",
        f"Probe games: {result['n_probe_games']}; ablation games: {result['n_ablation_games']}.",
        "",
        "| Layer | J variance | Full board acc. | J board acc. | Orth. board acc. | "
        "J legal P@5 | Orth. legal P@5 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for layer in layers:
        geometry = result["geometry"][str(layer)]
        probes = result["probes"][str(layer)]
        lines.append(
            f"| {layer} | {geometry['activation_variance_fraction']:.3f} | "
            f"{probes['full']['board_state_accuracy']:.3f} | "
            f"{probes['jspace']['board_state_accuracy']:.3f} | "
            f"{probes['orthogonal']['board_state_accuracy']:.3f} | "
            f"{probes['jspace']['legal_precision_at_5']:.3f} | "
            f"{probes['orthogonal']['legal_precision_at_5']:.3f} |"
        )
    lines += [
        "",
        "![Probe comparison](probe_accuracy_by_component.svg)",
        "",
        "![J-space variance](jspace_variance_by_layer.svg)",
        "",
        "![Ablation effects](ablation_effect_by_layer.svg)",
        "",
        "A workspace-like dissociation requires low J-space variance, strong action "
        "information, and disproportionate causal damage from J-space removal relative "
        "to matched random removal.",
        "",
    ]
    (out_dir / "workspace_split_summary.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint")
    parser.add_argument("--lens", default="out/coffee_v2/othello_jlens.pt")
    parser.add_argument("--n-probe-games", type=int, default=60)
    parser.add_argument("--n-ablation-games", type=int, default=20)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--skip-first", type=int, default=16)
    parser.add_argument("--probe-seed", type=int, default=3)
    parser.add_argument("--ablation-seed", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--out-dir", default="out/precoffee/workspace")
    args = parser.parse_args()
    result = run(args)
    write_artifacts(result, Path(args.out_dir))
    print(f"saved workspace-split artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()
