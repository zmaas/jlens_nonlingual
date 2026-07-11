from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


class FakeLens:
    jacobians = {0: torch.eye(4)}


def test_j_direction_matches_identity_transport_unembedding() -> None:
    from intervene_othello_jspace import _j_direction

    unembed = torch.eye(4)
    direction = _j_direction(FakeLens(), unembed, 0, 2, torch.device("cpu"))
    assert torch.allclose(direction, torch.tensor([0.0, 0.0, 1.0, 0.0]))
