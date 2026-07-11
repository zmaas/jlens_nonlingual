from __future__ import annotations

import xml.etree.ElementTree as ET

from precoffee_common import bootstrap_ci, line_chart_svg


def test_group_bootstrap_is_deterministic() -> None:
    groups = [[0.0, 1.0], [1.0, 1.0], [0.0]]
    first = bootstrap_ci(groups, n_bootstrap=100, seed=7)
    assert first == bootstrap_ci(groups, n_bootstrap=100, seed=7)
    assert 0 <= first[0] <= first[1] <= 1


def test_line_chart_is_valid_svg(tmp_path) -> None:
    path = tmp_path / "chart.svg"
    line_chart_svg(
        title="Test",
        y_label="Value",
        x_values=[0, 1, 2],
        series={"a": [0.1, 0.2, 0.3], "b": [0.3, 0.2, 0.1]},
        path=path,
        y_min=0,
        y_max=1,
    )
    ET.parse(path)
