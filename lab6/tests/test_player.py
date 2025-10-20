from src.Player import Player
from src.Deck import Card


import pytest


class TestPlayer:

    def test_player_initialization(self, sample_player):
        assert sample_player.name == "Test"
        assert sample_player.chips == 100
        assert sample_player.hand == []

    def test_player_deck(self, mock_players_deck, mock_card_king, mock_card_ace):
        assert mock_players_deck.hand[0] == mock_card_ace
        assert mock_players_deck.hand[1] == mock_card_king

    def test_reset_hand(self, mock_players_deck):
        mock_players_deck.reset_hand()
        assert mock_players_deck.hand == []

    def test_add_card(self, sample_player, mock_card_ace):
        sample_player.add_card(mock_card_ace)
        assert sample_player.hand[0] == mock_card_ace

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

    @pytest.mark.parametrize(
        "bet",
        [
            70,
            100,
            1
        ],
    )
    def test_make_bet_normal(self, sample_player, bet):
        assert bet == sample_player.make_bet(bet)

    @pytest.mark.parametrize("cards, expected_value", [
        (("Ace", "King"), 21),
        (("Ace", "9"), 20),
        (("Ace", "Ace", "9"), 21),
        (("10", "10", "2"), 22),
        (("Ace", "Ace", "Ace", "8"), 21),
    ])
    def test_hand_value_parametrized(self, sample_player, cards, expected_value):
        for rank in cards:
            sample_player.add_card(Card("Hearts", rank))

        assert sample_player.hand_value() == expected_value

class TestDealer:
    @pytest.mark.parametrize("cards, expected_value", [
        (("Ace", "King"), 21),
        (("Ace", "9"), 20),
        (("Ace", "Ace", "9"), 21),
        (("10", "10", "2"), 22),
        (("Ace", "Ace", "Ace", "8"), 21),
    ])
    def test_hand_value_parametrized(self, sample_dealer, cards, expected_value):
        for rank in cards:
            sample_dealer.add_card(Card("Hearts", rank))

        assert sample_dealer.should_hit() == expected_value
