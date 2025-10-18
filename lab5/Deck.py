import random
from log_module import module_logger, log_errors
from loguru import logger

class Card:
    """
    Класс реализующий карту
    """

    @log_errors(logger)
    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank

    @log_errors(logger)
    def value(self):
        """
        Метод определяет значение карты
        """
        if self.rank in ["Jack", "Queen", "King"]:
            return 10
        if self.rank == "Ace":
            return 11
        return int(self.rank)

    @log_errors(logger)
    def __str__(self):
        return self.rank + " of " + self.suit


class Deck:
    """
    Класс реализующий колоду
    """
    suits = ["Hearts", "Diamonds", "Clubs", "Spades"]
    ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace"]

    @log_errors(logger)
    def __init__(self):
        self.all_cards = [Card(suit, rank) for suit in self.suits for rank in self.ranks]
        self.shuffle()

    @log_errors(logger)
    def shuffle(self):
        """
        Метод перемешки колоды
        """
        random.shuffle(self.all_cards)

    @log_errors(logger)
    def deal_one(self):
        """
        Метод выдачи одной карты
        """
        if not self.all_cards:
            raise ValueError("Deck is empty")
        return self.all_cards.pop()
