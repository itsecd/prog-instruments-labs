from src.Player import Player


import pytest


class TestPlayer:




    @pytest.mark.parametrize(
        ("bet", "exception_str"),
        [
            (-50, "Bet must be"),
            (0, "Bet must be"),
            (100500, "Not enough")
        ],
    )
    def test_make_bet_negative(self, bet, exception_str, sample_player):
        with pytest.raises(ValueError) as exc:
            sample_player.make_bet(bet)

        assert exception_str in str(exc.value)

    def test_make_bet_normal(self, sample_player):
        bet = 70
        assert bet == sample_player.make_bet(bet)
