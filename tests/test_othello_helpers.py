from __future__ import annotations

import random

from othello_common import PASS_TOKEN, generate_games, random_game, token_label


def test_generated_games_fit_othellogpt_vocabulary() -> None:
    game = random_game(random.Random(0))
    assert 18 <= len(game) <= 59
    assert all(0 <= token <= PASS_TOKEN for token in game)
    assert token_label(0) == "A1"
    assert token_label(PASS_TOKEN) == "PASS"


def test_generation_is_deterministic_and_distinct() -> None:
    first = generate_games(3, seed=7)
    assert first == generate_games(3, seed=7)
    assert len({tuple(game) for game in first}) == 3
