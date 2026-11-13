"""Tests for tennis scoring."""

import pytest
from src.tennis import TennisGame


@pytest.mark.parametrize(
    ("first_points", "second_points", "expected"),
    [
        (0, 0, "Love-All"),
        (1, 1, "Fifteen-All"),
        (2, 2, "Thirty-All"),
        (3, 3, "Deuce"),
        (4, 4, "Deuce"),
        (1, 0, "Fifteen-Love"),
        (0, 2, "Love-Thirty"),
        (3, 2, "Forty-Thirty"),
        (4, 3, "Advantage Player 1"),
        (3, 4, "Advantage Player 2"),
        (5, 3, "Win for Player 1"),
        (2, 4, "Win for Player 2"),
    ],
)
def test_score(first_points: int, second_points: int, expected: str) -> None:
    game = TennisGame("Player 1", "Player 2")
    for _ in range(first_points):
        game.won_point("Player 1")
    for _ in range(second_points):
        game.won_point("Player 2")

    assert game.score() == expected


def test_unknown_player_is_rejected() -> None:
    game = TennisGame("Player 1", "Player 2")

    with pytest.raises(ValueError, match="Unknown player"):
        game.won_point("Player 3")
