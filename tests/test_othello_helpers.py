from __future__ import annotations

import random

from othello_common import (
    SQUARE_TO_TOKEN,
    TOKEN_TO_SQUARE,
    UNUSED_TOKEN,
    _captures,
    generate_games,
    legal_moves,
    random_game,
    token_label,
)


def test_generated_games_fit_othellogpt_vocabulary() -> None:
    game = random_game(random.Random(0))
    assert 18 <= len(game) <= 60
    assert all(1 <= token <= 60 for token in game)
    assert token_label(UNUSED_TOKEN) == "UNUSED"
    assert token_label(1) == "A1"
    assert token_label(60) == "H8"


def test_generation_is_deterministic_and_distinct() -> None:
    first = generate_games(3, seed=7)
    assert first == generate_games(3, seed=7)
    assert len({tuple(game) for game in first}) == 3


def test_generated_game_replays_as_legal_without_emitted_passes() -> None:
    game = random_game(random.Random(11), max_moves=60)
    board = [0] * 64
    board[27] = board[36] = -1
    board[28] = board[35] = 1
    player = 1
    for token in game:
        moves = legal_moves(board, player)
        if not moves:
            player = -player
            moves = legal_moves(board, player)
        square = TOKEN_TO_SQUARE[token - 1]
        assert square in moves
        captured = _captures(board, square, player)
        board[square] = player
        for captured_square in captured:
            board[captured_square] = player
        player = -player
    assert all(SQUARE_TO_TOKEN[TOKEN_TO_SQUARE[token - 1]] == token for token in game)


def test_position_states_align_targets_and_legal_moves() -> None:
    from othello_common import game_position_states

    game = random_game(random.Random(17), max_moves=60)
    records = game_position_states(game, skip_first=16)
    assert records
    for record in records:
        assert record["target"] == game[record["position"] + 1]
        assert record["target"] in record["legal_tokens"]
