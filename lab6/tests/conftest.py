import pytest
from src.Player import Player, Dealer
from src.Deck import Card

@pytest.fixture
def sample_player():
    return Player("Test", 100)