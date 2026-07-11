"""TransformerLens adapter for Anthropic's Jacobian Lens API."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn


class _TokenDecoder:
    """Minimal tokenizer surface used by jlens visualisation helpers."""

    def __init__(self, model: nn.Module) -> None:
        self._model = model

    def decode(self, token_ids: Sequence[int] | torch.Tensor, **_: Any) -> str:
        ids = [int(i) for i in token_ids]
        tokenizer = getattr(self._model, "tokenizer", None)
        if tokenizer is not None:
            return tokenizer.decode(ids)
        return " ".join(str(i) for i in ids)


class TransformerLensLensModel:
    """Wrap a loaded :class:`transformer_lens.HookedTransformer`.

    The wrapper holds references to the caller's model and downloads nothing.
    Integer lists/tensors are the normal Othello path. Strings are delegated to
    TransformerLens tokenisation and therefore require a configured tokenizer.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        self.layers = model.blocks
        self.n_layers = int(model.cfg.n_layers)
        self.d_model = int(model.cfg.d_model)
        self.tokenizer = _TokenDecoder(model)
        if len(self.layers) != self.n_layers:
            raise ValueError(
                f"cfg.n_layers={self.n_layers}, but model has {len(self.layers)} blocks"
            )

    @property
    def input_device(self) -> torch.device:
        return self.model.W_E.device

    def encode(self, prompt: Any, *, max_length: int = 512) -> torch.Tensor:
        if torch.is_tensor(prompt):
            tokens = prompt.detach()
        elif isinstance(prompt, str):
            tokens = self.model.to_tokens(prompt, prepend_bos=False)
        else:
            tokens = torch.as_tensor(prompt, dtype=torch.long)
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.ndim != 2 or tokens.shape[0] != 1:
            raise ValueError(
                f"prompt must encode to [seq] or [1, seq], got {tuple(tokens.shape)}"
            )
        return tokens[:, :max_length].to(device=self.input_device, dtype=torch.long)

    def forward(self, input_ids: torch.Tensor) -> Any:
        return self.model(input_ids, return_type=None)

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        residual = residual.to(device=self.input_device, dtype=self.model.W_U.dtype)
        if hasattr(self.model, "ln_final"):
            residual = self.model.ln_final(residual)
        return self.model.unembed(residual)
