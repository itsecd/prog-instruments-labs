from src.Player import Player


import pytest


def test_make_bet_negative():
    player = Player("Test", 100)
    bet = -50
    with pytest.raises(ValueError) as exc:
        player.make_bet(bet)

    assert "Bet must be" in str(exc.value)


def test_make_bet_zero():
    player = Player("Test", 100)
    bet = 0
    with pytest.raises(ValueError) as exc:
        player.make_bet(bet)

    assert "Bet must be" in str(exc.value)
