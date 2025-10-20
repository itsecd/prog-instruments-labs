import pytest
from src.Player import Player, Dealer
from src.Deck import Card


@pytest.fixture
def sample_player():
    return Player("Test", 100)

@pytest.fixture
def mock_card_king():
    return Card("Hearts", "King")

@pytest.fixture
def mock_card_ace():
    return Card("Spades", "Ace")

@pytest.fixture
def mock_players_deck(sample_player, mock_card_king, mock_card_ace):
    sample_player.add_card(mock_card_ace)
    sample_player.add_card(mock_card_king)
    return sample_player
