#!/usr/bin/env python3
"""OthelloGPT workspace closure: probes, rank sweeps, sparse cones, weak mode."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from othello_common import (  # noqa: E402
    CENTER_SQUARES,
    TOKEN_TO_SQUARE,
    game_position_states,
    generate_games,
    load_model,
    write_json,
)
from precoffee_common import (  # noqa: E402
    line_chart_svg,
    modify_block_output,
    summarize_grouped,
    validate_lens_metadata,
)

COMPONENT_LABELS = {
    "full": "Full residual stream",
    "jspace": "Linear J-span",
    "orthogonal": "Orthogonal remainder",
    "random_matched": "Matched random span",
}
BEHAVIOR_METRICS = (
    "legal_probability_mass",
    "legal_precision_at_5",
    "all_top_five_legal",
    "sampled_target_top_5_inclusion",
    "target_rank",
    "output_kl",
)


def _parse_ints(value: str) -> list[int]:
    result = [int(item) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected a comma-separated list of integers")
    return result


def _orthonormal_basis(matrix, *, relative_tolerance: float = 1e-5):
    import torch

    u, singular_values, vh = torch.linalg.svd(matrix.float(), full_matrices=False)
    threshold = singular_values[0] * relative_tolerance
    rank = int((singular_values > threshold).sum())
    return u[:, :rank], singular_values, vh


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
    absolute_boards, relative_boards = [], []
    legal_targets, next_targets, players, groups = [], [], [], []
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
            board = torch.tensor(record["board"], dtype=torch.long)
            player = int(record["player"])
            legal = torch.zeros(60)
            legal[[token - 1 for token in record["legal_tokens"]]] = 1
            absolute_boards.append(board)
            relative_boards.append(board * player)
            legal_targets.append(legal)
            next_targets.append(record["target"] - 1)
            players.append(player)
            groups.append(game_index)
    return {
        "activations": {layer: torch.cat(values) for layer, values in activations.items()},
        "absolute_board": torch.stack(absolute_boards),
        "relative_board": torch.stack(relative_boards),
        "legal": torch.stack(legal_targets),
        "next_move": torch.tensor(next_targets),
        "player": torch.tensor(players),
        "groups": torch.tensor(groups),
    }


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
    weights = torch.linalg.solve(train.T @ train + ridge * identity, train.T @ train_y)
    return test @ weights


def _balanced_accuracy(prediction, target, classes=(-1, 0, 1)) -> float:
    scores = []
    for value in classes:
        mask = target == value
        if bool(mask.any()):
            scores.append(float((prediction[mask] == value).float().mean()))
    return sum(scores) / len(scores)


def _probe_component(train_x, test_x, data, train_mask, test_mask, *, ridge: float):
    import torch

    result: dict[str, float] = {}
    for name in ("absolute_board", "relative_board"):
        target = data[name].to(train_x.device)
        train_target = target[train_mask]
        one_hot = torch.stack([(train_target == value).float() for value in (-1, 0, 1)], dim=-1)
        scores = _ridge_predict(
            train_x,
            test_x,
            one_hot.reshape(len(train_target), -1),
            ridge=ridge,
        ).reshape(-1, 64, 3)
        prediction = torch.tensor((-1, 0, 1), device=scores.device)[scores.argmax(dim=-1)]
        truth = target[test_mask]
        prefix = name.replace("_board", "_board_state")
        result[f"{prefix}_accuracy"] = float((prediction == truth).float().mean())
        result[f"{prefix}_balanced_accuracy"] = _balanced_accuracy(prediction, truth)
        occupied = truth != 0
        result[f"{prefix}_occupied_owner_accuracy"] = float(
            (prediction[occupied] == truth[occupied]).float().mean()
        )

    legal = data["legal"].to(train_x.device)
    legal_scores = _ridge_predict(train_x, test_x, legal[train_mask], ridge=ridge)
    legal_truth = legal[test_mask]
    legal_top = torch.topk(legal_scores, 5, dim=-1).indices
    legal_hits = legal_truth.gather(1, legal_top).sum(dim=1)
    result["legal_precision_at_5"] = float((legal_hits / 5).mean())
    result["legal_recall_at_5"] = float((legal_hits / legal_truth.sum(dim=1)).mean())

    player = data["player"].to(train_x.device).float()
    player_scores = _ridge_predict(train_x, test_x, player[train_mask, None], ridge=ridge)[:, 0]
    result["player_to_move_accuracy"] = float(
        ((player_scores > 0) == (player[test_mask] > 0)).float().mean()
    )

    next_move = data["next_move"].to(train_x.device)
    one_hot_next = torch.nn.functional.one_hot(next_move[train_mask], 60).float()
    next_scores = _ridge_predict(train_x, test_x, one_hot_next, ridge=ridge)
    next_top = torch.topk(next_scores, 5, dim=-1).indices
    result["sampled_target_top_5_inclusion"] = float(
        (next_top == next_move[test_mask, None]).any(dim=1).float().mean()
    )
    return result


def _final_logits(adapter, input_ids, positions: list[int]):
    from jlens import ActivationRecorder

    final_layer = adapter.n_layers - 1
    with ActivationRecorder(adapter.layers, at=[final_layer]) as recorder:
        adapter.forward(input_ids)
    residual = recorder.activations[final_layer][0, positions].detach().float()
    return adapter.unembed(residual).float().cpu()


def _intervened_logits(adapter, input_ids, positions, *, layer: int, transform: Callable):
    from jlens import ActivationRecorder

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


def _behavior_metrics(logits, records, baseline_logits=None):
    import torch

    values = defaultdict(list)
    for row, record in enumerate(records):
        legal = set(record["legal_tokens"])
        probabilities = torch.softmax(logits[row], dim=-1)
        top = torch.topk(logits[row], 5).indices.tolist()
        legal_count = sum(token in legal for token in top)
        values["legal_precision_at_5"].append(legal_count / 5)
        values["all_top_five_legal"].append(float(legal_count == 5))
        values["sampled_target_top_5_inclusion"].append(float(record["target"] in top))
        values["legal_probability_mass"].append(float(probabilities[list(legal)].sum()))
        values["target_rank"].append(float((logits[row] > logits[row, record["target"]]).sum() + 1))
        if baseline_logits is not None:
            baseline_prob = torch.softmax(baseline_logits[row], dim=-1)
            kl = (
                baseline_prob
                * (baseline_prob.clamp_min(1e-12).log() - probabilities.clamp_min(1e-12).log())
            ).sum()
            values["output_kl"].append(float(kl))
    return dict(values)


def _summarize_conditions(raw, *, n_bootstrap: int, seed: int):
    result = {}
    for condition, per_game in raw.items():
        result[condition] = {}
        metric_names = sorted({metric for game in per_game for metric in game})
        for metric_index, metric in enumerate(metric_names):
            groups = [game[metric] for game in per_game if metric in game]
            result[condition][metric] = summarize_grouped(
                groups,
                n_bootstrap=n_bootstrap,
                seed=seed + metric_index,
            )
    return result


def _prepare_behavior_games(adapter, games, *, skip_first: int):
    prepared = []
    for game in games:
        records = game_position_states(game, skip_first=skip_first)
        positions = [record["position"] for record in records]
        input_ids = adapter.encode(game, max_length=59)
        baseline = _final_logits(adapter, input_ids, positions)
        prepared.append((input_ids, positions, records, baseline))
    return prepared


def _baseline_behavior(prepared, *, n_bootstrap: int, seed: int):
    raw = {
        "baseline": [
            _behavior_metrics(baseline, records, baseline)
            for _input_ids, _positions, records, baseline in prepared
        ]
    }
    return _summarize_conditions(raw, n_bootstrap=n_bootstrap, seed=seed)["baseline"]


def _rank_sweep(adapter, prepared, bases, rank_values, *, n_bootstrap: int, seed: int):
    raw = defaultdict(list)
    for layer, basis in bases.items():
        valid_ranks = sorted({min(value, basis.shape[1]) for value in rank_values})
        for rank in valid_ranks:
            q = basis[:, :rank]
            for mode in ("keep", "remove"):
                condition = f"layer_{layer}/{mode}/rank_{rank}"
                for input_ids, positions, records, baseline in prepared:

                    def transform(tensor, q=q, mode=mode):
                        local_q = q.to(tensor)
                        projection = (tensor @ local_q) @ local_q.T
                        return projection if mode == "keep" else tensor - projection

                    logits = _intervened_logits(
                        adapter, input_ids, positions, layer=layer, transform=transform
                    )
                    raw[condition].append(_behavior_metrics(logits, records, baseline))
    return _summarize_conditions(raw, n_bootstrap=n_bootstrap, seed=seed)


def _nonnegative_pursuit(x, dictionary, *, max_support: int, refit_steps: int):
    """Approximate batched nonnegative gradient pursuit on unit-norm atoms."""
    import torch

    shape = x.shape
    flat = x.reshape(-1, shape[-1]).float()
    atoms = dictionary.float()
    atoms = atoms / atoms.norm(dim=0, keepdim=True).clamp_min(1e-8)
    gram = atoms.T @ atoms
    targets = flat @ atoms
    step_size = float(0.95 / torch.linalg.eigvalsh(gram).max().clamp_min(1e-6))
    coeffs = torch.zeros(flat.shape[0], atoms.shape[1], device=flat.device)
    selected = torch.zeros_like(coeffs, dtype=torch.bool)
    snapshots = {}
    requested = set(range(1, max_support + 1))
    for support in range(1, max_support + 1):
        residual = flat - coeffs @ atoms.T
        scores = residual @ atoms
        scores[selected] = -torch.inf
        best_score, best = scores.max(dim=1)
        active = best_score > 0
        selected[active, best[active]] = True
        for _ in range(refit_steps):
            gradient = coeffs @ gram - targets
            coeffs = torch.relu(coeffs - step_size * gradient) * selected
        if support in requested:
            snapshots[support] = (
                (coeffs @ atoms.T).reshape(shape).to(x.dtype),
                coeffs.reshape(*shape[:-1], -1),
            )
    return snapshots


def _sparse_readout(data, directions, support_values, *, refit_steps: int):
    output = {}
    legal = data["legal"]
    target = data["next_move"]
    for layer, x_cpu in data["activations"].items():
        x = x_cpu.to(directions[layer].device)
        snapshots = _nonnegative_pursuit(
            x,
            directions[layer],
            max_support=max(support_values),
            refit_steps=refit_steps,
        )
        layer_result = {}
        centered_energy = ((x - x.mean(dim=0, keepdim=True)) ** 2).sum().clamp_min(1e-8)
        for support in support_values:
            reconstruction, coeffs = snapshots[support]
            active = coeffs > 1e-7
            legal_local = legal.to(active.device).bool()
            target_local = target.to(active.device)
            active_count = active.sum(dim=1).clamp_min(1)
            selected_legal = (active & legal_local).sum(dim=1)
            layer_result[str(support)] = {
                "reconstruction_r_squared": float(
                    1 - ((x - reconstruction) ** 2).sum() / centered_energy
                ),
                "mean_support_size": float(active.sum(dim=1).float().mean()),
                "selected_atom_legal_precision": float(
                    (selected_legal / active_count).float().mean()
                ),
                "selected_atom_legal_recall": float(
                    (selected_legal / legal_local.sum(dim=1).clamp_min(1)).float().mean()
                ),
                "sampled_target_selected": float(
                    active.gather(1, target_local[:, None]).float().mean()
                ),
            }
        output[str(layer)] = layer_result
    return output


def _sparse_causal_sweep(
    adapter,
    prepared,
    directions,
    layers,
    support_values,
    *,
    refit_steps: int,
    n_bootstrap: int,
    seed: int,
):
    raw = defaultdict(list)
    for layer in layers:
        dictionary = directions[layer]
        for support in support_values:
            for mode in ("keep", "remove"):
                condition = f"layer_{layer}/{mode}/support_{support}"
                for input_ids, positions, records, baseline in prepared:

                    def transform(
                        tensor,
                        support=support,
                        mode=mode,
                        dictionary=dictionary,
                    ):
                        component = _nonnegative_pursuit(
                            tensor,
                            dictionary.to(tensor),
                            max_support=support,
                            refit_steps=refit_steps,
                        )[support][0]
                        return component if mode == "keep" else tensor - component

                    logits = _intervened_logits(
                        adapter, input_ids, positions, layer=layer, transform=transform
                    )
                    raw[condition].append(_behavior_metrics(logits, records, baseline))
    return _summarize_conditions(raw, n_bootstrap=n_bootstrap, seed=seed)


def _weak_mode_diagnostics(directions, full_directions, move_unembed, full_unembed):
    import torch

    output = {}
    uniform = torch.ones(move_unembed.shape[1], device=move_unembed.device)
    uniform = uniform / uniform.norm()
    centered_unembed = move_unembed - move_unembed.mean(dim=1, keepdim=True)
    raw_singular = torch.linalg.svdvals(move_unembed.float())
    centered_singular = torch.linalg.svdvals(centered_unembed.float())
    full_singular = torch.linalg.svdvals(full_unembed.float())
    for layer, matrix in directions.items():
        _, singular, vh = torch.linalg.svd(matrix.float(), full_matrices=False)
        centered_singular = torch.linalg.svdvals(
            (matrix - matrix.mean(dim=1, keepdim=True)).float()
        )
        full_token_singular = torch.linalg.svdvals(full_directions[layer].float())
        weak = vh[-1]
        output[str(layer)] = {
            "singular_values": singular.cpu().tolist(),
            "centered_move_dictionary_singular_values": centered_singular.cpu().tolist(),
            "full_61_token_dictionary_singular_values": full_token_singular.cpu().tolist(),
            "weak_to_next_ratio": float(singular[-1] / singular[-2]),
            "next_to_weak_gap": float(singular[-2] / singular[-1].clamp_min(1e-12)),
            "weak_mode_uniform_abs_cosine": float(torch.abs(weak @ uniform)),
            "weak_mode_coefficients": weak.cpu().tolist(),
            "weak_mode_coefficient_std": float(weak.std()),
            "weak_mode_coefficient_min": float(weak.min()),
            "weak_mode_coefficient_max": float(weak.max()),
        }
    output["controls"] = {
        "move_unembedding_singular_values": raw_singular.cpu().tolist(),
        "centered_move_unembedding_singular_values": centered_singular.cpu().tolist(),
        "full_61_token_unembedding_singular_values": full_singular.cpu().tolist(),
    }
    return output


def _make_transparent(path: Path) -> None:
    text = path.read_text()
    text = text.replace('<rect width="720" height="360" fill="#ffffff"/>\n', "")
    path.write_text(text)


def _line_figure(*, path: Path, **kwargs) -> None:
    line_chart_svg(path=path, **kwargs)
    _make_transparent(path)


def _heatmap_svg(coefficients, path: Path, *, title: str) -> None:
    maximum = max(abs(value) for value in coefficients) or 1.0
    rows = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="650" height="590" viewBox="0 0 650 590">',
        '<text x="40" y="28" font-family="system-ui" font-size="17" '
        f'font-weight="600" fill="#d9e2f2">{title}</text>',
    ]
    values_by_square = {square: coefficients[i] for i, square in enumerate(TOKEN_TO_SQUARE)}
    for square in range(64):
        row, col = divmod(square, 8)
        x, y = 52 + col * 64, 52 + row * 64
        if square in CENTER_SQUARES:
            fill, label = "#273142", "start"
        else:
            value = values_by_square[square]
            strength = min(1.0, abs(value) / maximum)
            if value >= 0:
                red = int(45 + 35 * (1 - strength))
                green = int(100 + 80 * (1 - strength))
                blue = int(210 + 35 * (1 - strength))
            else:
                red = int(220 + 25 * (1 - strength))
                green = int(65 + 100 * (1 - strength))
                blue = int(75 + 100 * (1 - strength))
            fill = f"rgb({red},{green},{blue})"
            label = f"{value:+.2f}"
        rows.append(f'<rect x="{x}" y="{y}" width="60" height="60" rx="5" fill="{fill}"/>')
        rows.append(
            f'<text x="{x + 30}" y="{y + 35}" text-anchor="middle" '
            f'font-family="system-ui" font-size="11" fill="#f6f8fa">{label}</text>'
        )
    for index in range(8):
        rows.append(
            f'<text x="{82 + index * 64}" y="580" text-anchor="middle" '
            f'font-family="system-ui" font-size="12" fill="#9eadc5">'
            f"{chr(65 + index)}</text>"
        )
        rows.append(
            f'<text x="35" y="{88 + index * 64}" text-anchor="middle" '
            f'font-family="system-ui" font-size="12" fill="#9eadc5">'
            f"{index + 1}</text>"
        )
    rows.append("</svg>")
    path.write_text("\n".join(rows) + "\n")


def _figures(results: dict[str, Any]) -> None:
    figures = EXPERIMENT_DIR / "figures"
    figures.mkdir(exist_ok=True)
    layers = [int(layer) for layer in results["probe_results"]]
    series = {}
    for component, label in COMPONENT_LABELS.items():
        series[label] = [
            results["probe_results"][str(layer)][component][
                "relative_board_state_balanced_accuracy"
            ]
            for layer in layers
        ]
    _line_figure(
        title="Relative board state decodable from each residual component",
        y_label="Balanced accuracy across opponent, empty, current player",
        x_values=layers,
        series=series,
        path=figures / "relative_board_state_by_layer.svg",
        y_min=0,
        y_max=1,
        percent=True,
    )

    final_layer = max(layers)
    baseline_mass = results["baseline_behavior"]["legal_probability_mass"]["mean"]
    rank_conditions = results.get("rank_sweep", {})
    if rank_conditions:
        ranks = sorted(
            int(key.rsplit("_", 1)[1])
            for key in rank_conditions
            if key.startswith(f"layer_{final_layer}/keep/rank_")
        )
        _line_figure(
            title=f"Layer {final_layer}: causal effect of linear J-span rank",
            y_label="Probability assigned to legal moves",
            x_values=ranks,
            series={
                "Keep leading J directions": [
                    rank_conditions[f"layer_{final_layer}/keep/rank_{rank}"][
                        "legal_probability_mass"
                    ]["mean"]
                    for rank in ranks
                ],
                "Remove leading J directions": [
                    rank_conditions[f"layer_{final_layer}/remove/rank_{rank}"][
                        "legal_probability_mass"
                    ]["mean"]
                    for rank in ranks
                ],
                "Unmodified model": [baseline_mass for _ in ranks],
            },
            path=figures / "rank_sweep_final_layer.svg",
            y_min=0,
            y_max=1,
            percent=True,
        )

    sparse_conditions = results.get("sparse_causal_sweep", {})
    if sparse_conditions:
        sparse_layers = sorted({int(key.split("/")[0].split("_")[1]) for key in sparse_conditions})
        layer = max(sparse_layers)
        supports = sorted(
            int(key.rsplit("_", 1)[1])
            for key in sparse_conditions
            if key.startswith(f"layer_{layer}/keep/support_")
        )
        _line_figure(
            title=f"Layer {layer}: sparse nonnegative J reconstruction",
            y_label="Probability assigned to legal moves",
            x_values=supports,
            series={
                "Keep sparse J component": [
                    sparse_conditions[f"layer_{layer}/keep/support_{support}"][
                        "legal_probability_mass"
                    ]["mean"]
                    for support in supports
                ],
                "Remove sparse J component": [
                    sparse_conditions[f"layer_{layer}/remove/support_{support}"][
                        "legal_probability_mass"
                    ]["mean"]
                    for support in supports
                ],
                "Unmodified model": [baseline_mass for _ in supports],
            },
            path=figures / "sparse_sweep_final_layer.svg",
            y_min=0,
            y_max=1,
            percent=True,
        )

    weak = results["weak_mode"][str(final_layer)]["weak_mode_coefficients"]
    _heatmap_svg(
        weak,
        figures / "weak_mode_final_layer.svg",
        title=f"Layer {final_layer} weakest J-space token combination",
    )


def _write_report(results: dict[str, Any], args) -> None:
    final_layer = str(max(int(layer) for layer in results["probe_results"]))
    full = results["probe_results"][final_layer]["full"]
    jspace = results["probe_results"][final_layer]["jspace"]
    orthogonal = results["probe_results"][final_layer]["orthogonal"]
    weak = results["weak_mode"][final_layer]
    n_probe = results["config"]["n_probe_games"]
    n_ablation = results["config"]["n_ablation_games"]
    full_relative = full["relative_board_state_balanced_accuracy"]
    jspace_relative = jspace["relative_board_state_balanced_accuracy"]
    orthogonal_relative = orthogonal["relative_board_state_balanced_accuracy"]
    weak_ratio = weak["weak_to_next_ratio"]
    uniform_cosine = weak["weak_mode_uniform_abs_cosine"]
    text = f"""# Othello workspace closure experiment

## Question

Does OthelloGPT expose a compact, causally useful move workspace, rather than a
60-dimensional subspace that merely inherits information from the output vocabulary?

## Run summary

- Probe games: {n_probe}; causal games: {n_ablation}.
- Final tested layer: {final_layer}.
- Relative board balanced accuracy: full residual **{full_relative:.3f}**, linear
  J-span **{jspace_relative:.3f}**, orthogonal remainder **{orthogonal_relative:.3f}**.
- Weakest/next-smallest singular-value ratio: **{weak_ratio:.3f}**.
- Weak mode absolute cosine with the uniform-token direction: **{uniform_cosine:.3f}**.

These values are descriptive. Interpret workspace claims primarily through the
causal rank and sparse sweeps in `data/results.json`; in particular, look for a
small rank/support that preserves legal probability mass when kept and damages
it when removed.

## Metrics

Board state uses balanced accuracy across opponent, empty, and current-player
labels. `legal_precision_at_5` is the fraction of the five highest-logit moves
that are legal. `sampled_target_top_5_inclusion` asks whether the particular
randomly sampled continuation lies in the top five; it is not pass@5.

## Method caveat

The sparse cone uses a documented local approximation: unit-normalized J atoms,
greedy positive-residual atom selection, and projected-gradient nonnegative
refitting on the selected support ({args.pursuit_refit_steps} iterations per
step). The vendored public J-lens repository does not include Anthropic's
gradient-pursuit implementation, so this result should not be presented as an
exact reproduction of their optimizer.

## Artifacts

- `data/results.json`: all metrics and spectra
- `figures/relative_board_state_by_layer.svg`
- `figures/rank_sweep_final_layer.svg`
- `figures/sparse_sweep_final_layer.svg`
- `figures/weak_mode_final_layer.svg`
- `logs/runtime.json`
"""
    (EXPERIMENT_DIR / "report.md").write_text(text)


def run(args) -> dict[str, Any]:
    import torch
    from jlens import JacobianLens
    from jlens.adapters import TransformerLensLensModel

    started = time.time()
    for subdir in ("data", "figures", "logs", "artifacts"):
        (EXPERIMENT_DIR / subdir).mkdir(exist_ok=True)

    if args.quick:
        args.n_probe_games = min(args.n_probe_games, 10)
        args.n_ablation_games = min(args.n_ablation_games, 2)
        args.rank_values = [4, 16, 60]
        args.sparse_values = [4, 12]
        args.sparse_causal_layers = [max(args.sparse_causal_layers)]
        args.n_bootstrap = min(args.n_bootstrap, 100)
        args.pursuit_refit_steps = min(args.pursuit_refit_steps, 10)

    metadata = validate_lens_metadata(args.lens)
    lens = JacobianLens.load(args.lens)
    adapter = TransformerLensLensModel(load_model(args.device, args.checkpoint))
    layers = list(lens.source_layers)
    invalid_sparse = set(args.sparse_causal_layers) - set(layers)
    if invalid_sparse:
        raise ValueError(f"sparse causal layers absent from lens: {sorted(invalid_sparse)}")

    full_unembed = adapter.model.W_U.detach().float().to(adapter.input_device)
    move_unembed = full_unembed[:, 1:]
    directions, full_directions, bases, geometry = {}, {}, {}, {}
    for layer in layers:
        jacobian = lens.jacobians[layer].to(adapter.input_device)
        matrix = jacobian.T @ move_unembed
        full_directions[layer] = jacobian.T @ full_unembed
        q, singular, _ = _orthonormal_basis(matrix)
        directions[layer] = matrix
        bases[layer] = q
        normalized = singular / singular.sum()
        effective_rank = float(torch.exp(-(normalized * normalized.clamp_min(1e-12).log()).sum()))
        geometry[str(layer)] = {
            "numerical_rank": q.shape[1],
            "effective_rank": effective_rank,
            "singular_values": singular.cpu().tolist(),
        }

    probe_games = generate_games(
        args.n_probe_games, seed=args.probe_seed, min_length=args.skip_first + 2
    )
    data = _collect_probe_data(adapter, probe_games, layers, skip_first=args.skip_first)
    n_train_games = max(1, int(args.train_fraction * args.n_probe_games))
    train_mask_cpu = data["groups"] < n_train_games
    test_mask_cpu = ~train_mask_cpu
    probe_results = {}
    for layer in layers:
        x = data["activations"][layer].to(adapter.input_device)
        train_mask = train_mask_cpu.to(x.device)
        test_mask = test_mask_cpu.to(x.device)
        q = bases[layer]
        random_q = _random_basis(
            adapter.d_model,
            q.shape[1],
            seed=args.random_seed + layer,
            device=x.device,
        )
        centered = x - x.mean(dim=0, keepdim=True)
        components = {
            "full": x,
            "jspace": (x @ q) @ q.T,
            "orthogonal": centered - (centered @ q) @ q.T,
            "random_matched": (x @ random_q) @ random_q.T,
        }
        probe_results[str(layer)] = {}
        for name, component in components.items():
            probe_results[str(layer)][name] = _probe_component(
                component[train_mask],
                component[test_mask],
                data,
                train_mask,
                test_mask,
                ridge=args.ridge,
            )

    sparse_readout = _sparse_readout(
        data,
        directions,
        args.sparse_values,
        refit_steps=args.pursuit_refit_steps,
    )
    ablation_games = generate_games(
        args.n_ablation_games,
        seed=args.ablation_seed,
        min_length=args.skip_first + 2,
    )
    prepared = _prepare_behavior_games(adapter, ablation_games, skip_first=args.skip_first)
    baseline_behavior = _baseline_behavior(
        prepared,
        n_bootstrap=args.n_bootstrap,
        seed=args.ablation_seed,
    )
    rank_sweep = {}
    if not args.skip_rank_causal:
        rank_sweep = _rank_sweep(
            adapter,
            prepared,
            bases,
            args.rank_values,
            n_bootstrap=args.n_bootstrap,
            seed=args.ablation_seed,
        )
    sparse_causal = {}
    if not args.skip_sparse_causal:
        sparse_causal = _sparse_causal_sweep(
            adapter,
            prepared,
            directions,
            args.sparse_causal_layers,
            args.sparse_values,
            refit_steps=args.pursuit_refit_steps,
            n_bootstrap=args.n_bootstrap,
            seed=args.ablation_seed + 1000,
        )

    result = {
        "experiment": "othello-workspace-closure",
        "config": {
            "device": args.device,
            "checkpoint": args.checkpoint,
            "lens": args.lens,
            "n_probe_games": args.n_probe_games,
            "n_ablation_games": args.n_ablation_games,
            "skip_first": args.skip_first,
            "rank_values": args.rank_values,
            "sparse_values": args.sparse_values,
            "sparse_causal_layers": args.sparse_causal_layers,
            "pursuit_refit_steps": args.pursuit_refit_steps,
            "quick": args.quick,
        },
        "lens_metadata": metadata,
        "geometry": geometry,
        "probe_results": probe_results,
        "sparse_readout": sparse_readout,
        "baseline_behavior": baseline_behavior,
        "rank_sweep": rank_sweep,
        "sparse_causal_sweep": sparse_causal,
        "weak_mode": _weak_mode_diagnostics(
            directions,
            full_directions,
            move_unembed,
            full_unembed,
        ),
    }
    write_json(EXPERIMENT_DIR / "data" / "results.json", result)
    _figures(result)
    _write_report(result, args)
    runtime = {
        "elapsed_seconds": time.time() - started,
        "torch_version": torch.__version__,
        "device": args.device,
        "completed": True,
    }
    write_json(EXPERIMENT_DIR / "logs" / "runtime.json", runtime)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", help="Local synthetic_model.pth; avoids a Hub download")
    parser.add_argument("--lens", default="out/coffee_v2/othello_jlens.pt")
    parser.add_argument("--n-probe-games", type=int, default=60)
    parser.add_argument("--n-ablation-games", type=int, default=20)
    parser.add_argument("--skip-first", type=int, default=16)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--probe-seed", type=int, default=41)
    parser.add_argument("--ablation-seed", type=int, default=43)
    parser.add_argument("--random-seed", type=int, default=47)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument(
        "--rank-values",
        type=_parse_ints,
        default=_parse_ints("1,2,4,8,16,32,59,60"),
    )
    parser.add_argument(
        "--sparse-values",
        type=_parse_ints,
        default=_parse_ints("1,2,4,8,12,16,25"),
    )
    parser.add_argument(
        "--sparse-causal-layers",
        type=_parse_ints,
        default=_parse_ints("4,5,6"),
    )
    parser.add_argument("--pursuit-refit-steps", type=int, default=25)
    parser.add_argument("--skip-rank-causal", action="store_true")
    parser.add_argument("--skip-sparse-causal", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run(args)
    final_layer = str(max(int(layer) for layer in result["probe_results"]))
    summary = {
        "results": str(EXPERIMENT_DIR / "data" / "results.json"),
        "report": str(EXPERIMENT_DIR / "report.md"),
        "final_layer_relative_board_balanced_accuracy": {
            name: values["relative_board_state_balanced_accuracy"]
            for name, values in result["probe_results"][final_layer].items()
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
