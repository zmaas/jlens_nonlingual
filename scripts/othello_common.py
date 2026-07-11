"""Shared OthelloGPT loading, game generation, and reporting helpers."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDORED_JLENS = REPO_ROOT / "vendor" / "jacobian-lens"
if str(VENDORED_JLENS) not in sys.path:
    sys.path.insert(0, str(VENDORED_JLENS))

CENTER_SQUARES = {27, 28, 35, 36}
TOKEN_TO_SQUARE = [square for square in range(64) if square not in CENTER_SQUARES]
SQUARE_TO_TOKEN = {square: token for token, square in enumerate(TOKEN_TO_SQUARE)}
PASS_TOKEN = 60
CHECKPOINT_REPO = "NeelNanda/Othello-GPT-Transformer-Lens"
CHECKPOINT_FILE = "synthetic_model.pth"
CAVEAT = (
    "This direct J-lens decodes move tokens, not board-state labels. Legal-move "
    "enrichment is suggestive only; board occupancy requires a probe/template extension."
)

_DIRECTIONS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _captures(board: list[int], square: int, player: int) -> list[int]:
    if board[square] != 0:
        return []
    row, col = divmod(square, 8)
    captured: list[int] = []
    for dr, dc in _DIRECTIONS:
        r, c, line = row + dr, col + dc, []
        while 0 <= r < 8 and 0 <= c < 8 and board[8 * r + c] == -player:
            line.append(8 * r + c)
            r, c = r + dr, c + dc
        if line and 0 <= r < 8 and 0 <= c < 8 and board[8 * r + c] == player:
            captured.extend(line)
    return captured


def legal_moves(board: list[int], player: int) -> list[int]:
    return [square for square in range(64) if _captures(board, square, player)]


def random_game(rng: random.Random, *, max_moves: int = 59) -> list[int]:
    """Generate a legal game in OthelloGPT's 61-token vocabulary."""
    board = [0] * 64
    board[27] = board[36] = -1
    board[28] = board[35] = 1
    player, passes = 1, 0
    tokens: list[int] = []
    while len(tokens) < max_moves and passes < 2:
        moves = legal_moves(board, player)
        if not moves:
            tokens.append(PASS_TOKEN)
            passes += 1
            player = -player
            continue
        passes = 0
        square = rng.choice(moves)
        captured = _captures(board, square, player)
        board[square] = player
        for captured_square in captured:
            board[captured_square] = player
        tokens.append(SQUARE_TO_TOKEN[square])
        player = -player
    return tokens


def generate_games(n_games: int, *, seed: int, min_length: int = 18) -> list[list[int]]:
    rng = random.Random(seed)
    games: list[list[int]] = []
    while len(games) < n_games:
        game = random_game(rng)
        if len(game) >= min_length:
            games.append(game)
    return games


def token_label(token: int) -> str:
    if token == PASS_TOKEN:
        return "PASS"
    if not 0 <= token < len(TOKEN_TO_SQUARE):
        return f"token-{token}"
    row, col = divmod(TOKEN_TO_SQUARE[token], 8)
    return f"{chr(ord('A') + col)}{row + 1}"


def parse_layers(value: str) -> list[int]:
    layers = [int(item) for item in value.split(",") if item.strip()]
    if not layers:
        raise ValueError("at least one source layer is required")
    return layers


def load_model(device: str, checkpoint_path: str | None = None):
    """Load OthelloGPT. A Hub download occurs only when no local path is given."""
    import torch
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    if checkpoint_path is None:
        from huggingface_hub import hf_hub_download

        checkpoint_path = hf_hub_download(CHECKPOINT_REPO, CHECKPOINT_FILE)
    cfg = HookedTransformerConfig(
        n_layers=8,
        d_model=512,
        d_head=64,
        n_heads=8,
        d_mlp=2048,
        d_vocab=61,
        n_ctx=59,
        act_fn="gelu",
        normalization_type="LNPre",
        device=device,
    )
    model = HookedTransformer(cfg)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
