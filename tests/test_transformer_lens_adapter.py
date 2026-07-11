from __future__ import annotations

# ruff: noqa: E402, I001

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn

import jlens
from jlens.adapters import TransformerLensLensModel


class Block(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, d_model, bias=False)

    def forward(self, residual):
        return residual + 0.05 * self.linear(residual)


class FakeHookedTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cfg = SimpleNamespace(n_layers=2, d_model=8)
        self.embed = nn.Embedding(16, 8)
        self.blocks = nn.ModuleList([Block(8), Block(8)])
        self.ln_final = nn.LayerNorm(8)
        self.unembed_module = nn.Linear(8, 16, bias=False)

    @property
    def W_E(self):
        return self.embed.weight

    @property
    def W_U(self):
        return self.unembed_module.weight.T

    def forward(self, tokens, return_type=None):
        residual = self.embed(tokens)
        for block in self.blocks:
            residual = block(residual)
        return None if return_type is None else self.unembed(residual)

    def unembed(self, residual):
        return self.unembed_module(residual)

    def to_tokens(self, text, prepend_bos=False):
        return torch.tensor([[ord(char) % 16 for char in text]])


def test_encode_accepts_string_list_and_tensor() -> None:
    adapter = TransformerLensLensModel(FakeHookedTransformer())
    assert adapter.encode("abcd", max_length=3).shape == (1, 3)
    assert adapter.encode([1, 2, 3]).tolist() == [[1, 2, 3]]
    assert adapter.encode(torch.tensor([[4, 5]])).tolist() == [[4, 5]]


def test_fit_and_apply_on_fake_transformer_lens_model() -> None:
    adapter = TransformerLensLensModel(FakeHookedTransformer())
    lens = jlens.fit(
        adapter,
        [[1, 2, 3, 4, 5], torch.tensor([5, 4, 3, 2, 1])],
        source_layers=[0],
        target_layer=1,
        dim_batch=4,
        max_seq_len=5,
        skip_first=0,
    )
    logits, final_logits, ids = lens.apply(adapter, [1, 2, 3, 4], positions=[-1])
    assert lens.n_prompts == 2
    assert logits[0].shape == (1, 16)
    assert final_logits.shape == (1, 16)
    assert ids.tolist() == [[1, 2, 3, 4]]
