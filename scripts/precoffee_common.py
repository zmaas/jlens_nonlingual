"""Shared metrics, plotting, and intervention helpers for pre-coffee runs."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from othello_common import TOKEN_ENCODING


def validate_lens_metadata(lens_path: str | Path) -> dict[str, Any]:
    path = Path(lens_path).with_suffix(".metadata.json")
    if not path.exists():
        raise RuntimeError(f"missing lens metadata: {path}")
    metadata = json.loads(path.read_text())
    if metadata.get("token_encoding") != TOKEN_ENCODING:
        raise RuntimeError(
            f"expected token encoding {TOKEN_ENCODING!r}, "
            f"found {metadata.get('token_encoding')!r}; refit the lens"
        )
    return metadata


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def bootstrap_ci(
    per_group_values: Sequence[Sequence[float]],
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile CI resampling groups while pooling observations within them."""
    if not per_group_values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    estimates = []
    n_groups = len(per_group_values)
    group_summaries = [(sum(group), len(group)) for group in per_group_values]
    for _ in range(n_bootstrap):
        total = 0.0
        count = 0
        for _ in range(n_groups):
            group_sum, group_count = group_summaries[rng.randrange(n_groups)]
            total += group_sum
            count += group_count
        estimates.append(total / count)
    estimates.sort()
    lo = estimates[int(0.025 * (n_bootstrap - 1))]
    hi = estimates[int(0.975 * (n_bootstrap - 1))]
    return lo, hi


def summarize_grouped(
    per_group_values: Sequence[Sequence[float]], *, n_bootstrap: int, seed: int
) -> dict[str, float | int]:
    pooled = [value for group in per_group_values for value in group]
    lo, hi = bootstrap_ci(per_group_values, n_bootstrap=n_bootstrap, seed=seed)
    return {
        "mean": mean(pooled),
        "ci95_low": lo,
        "ci95_high": hi,
        "n": len(pooled),
        "n_groups": len(per_group_values),
    }


def line_chart_svg(
    *,
    title: str,
    y_label: str,
    x_values: Sequence[int],
    series: dict[str, Sequence[float]],
    path: str | Path,
    y_min: float | None = None,
    y_max: float | None = None,
    percent: bool = False,
) -> None:
    """Write a dependency-free compact SVG line chart."""
    colors = ["#0969da", "#bf3989", "#1a7f37", "#9a6700", "#8250df"]
    left, right, top, bottom = 68, 690, 38, 302
    all_values = [value for values in series.values() for value in values]
    lo = min(all_values) if y_min is None else y_min
    hi = max(all_values) if y_max is None else y_max
    if math.isclose(lo, hi):
        hi = lo + 1.0

    def x_coord(index: int) -> float:
        return left + index * (right - left) / max(1, len(x_values) - 1)

    def y_coord(value: float) -> float:
        return bottom - (value - lo) * (bottom - top) / (hi - lo)

    def label(value: float) -> str:
        return f"{100 * value:.0f}%" if percent else f"{value:.2f}"

    rows = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="360" '
        'viewBox="0 0 720 360" role="img">',
        '<rect width="720" height="360" fill="#ffffff"/>',
        f'<text x="68" y="20" font-family="system-ui" font-size="16" '
        f'font-weight="600" fill="#24292f">{title}</text>',
    ]
    for tick in range(5):
        value = lo + tick * (hi - lo) / 4
        y = y_coord(value)
        rows.append(f'<path d="M{left} {y:.1f}H{right}" stroke="#d8dee4"/>')
        rows.append(
            f'<text x="58" y="{y + 4:.1f}" text-anchor="end" '
            f'font-family="system-ui" font-size="11" fill="#57606a">'
            f"{label(value)}</text>"
        )
    rows.append(f'<path d="M{left} {top}V{bottom}H{right}" fill="none" stroke="#57606a"/>')
    for index, value in enumerate(x_values):
        x = x_coord(index)
        rows.append(
            f'<text x="{x:.1f}" y="322" text-anchor="middle" '
            f'font-family="system-ui" font-size="11" fill="#57606a">{value}</text>'
        )
    rows.append(
        f'<text x="18" y="170" transform="rotate(-90 18 170)" '
        f'text-anchor="middle" font-family="system-ui" font-size="11" '
        f'fill="#24292f">{y_label}</text>'
    )
    for series_index, (name, values) in enumerate(series.items()):
        color = colors[series_index % len(colors)]
        points = " ".join(
            f"{x_coord(i):.1f},{y_coord(value):.1f}" for i, value in enumerate(values)
        )
        rows.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        for i, value in enumerate(values):
            rows.append(
                f'<circle cx="{x_coord(i):.1f}" cy="{y_coord(value):.1f}" r="4" fill="{color}"/>'
            )
        legend_x = 75 + series_index * 125
        rows.append(
            f'<line x1="{legend_x}" y1="344" x2="{legend_x + 24}" y2="344" '
            f'stroke="{color}" stroke-width="3"/>'
        )
        rows.append(
            f'<text x="{legend_x + 30}" y="348" font-family="system-ui" '
            f'font-size="11" fill="#24292f">{name}</text>'
        )
    rows.append("</svg>")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n")


def modify_block_output(output, transform: Callable):
    """Apply ``transform`` to a tensor block output, preserving tuple outputs."""
    import torch

    if torch.is_tensor(output):
        return transform(output)
    return (transform(output[0]), *output[1:])
