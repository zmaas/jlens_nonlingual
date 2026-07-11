from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_jspace_basis_is_orthonormal_and_detects_rank() -> None:
    from analyze_othello_workspace_split import _orthonormal_basis

    matrix = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    basis, _ = _orthonormal_basis(matrix)
    assert basis.shape == (3, 2)
    assert torch.allclose(basis.T @ basis, torch.eye(2), atol=1e-5)
