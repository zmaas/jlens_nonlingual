from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_position_legality_metrics() -> None:
    from eval_othello_legality import _position_metrics

    logits = torch.arange(61, dtype=torch.float32)
    metrics = _position_metrics(logits, [58, 59, 60], target=60, k=5)
    assert metrics["legal_precision_at_k"] == 3 / 5
    assert metrics["legal_recall_at_k"] == 1
    assert metrics["any_legal_at_k"] == 1
    assert metrics["unused_token_at_k"] == 0
    assert metrics["target_pass_at_k"] == 1
