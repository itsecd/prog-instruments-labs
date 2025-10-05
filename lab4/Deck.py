import random


class Card:
    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank

    def value(self):
        if self.rank in ["Jack", "Queen", "King"]:
            return 10
        if self.rank == "Ace":
            return 11
        return int(self.rank)

    def __str__(self):
        return self.rank + " of " + self.suit


class Deck:
    suits = ["Hearts", "Diamonds", "Clubs", "Spades"]
    ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace"]

    def __init__(self):
        self.all_cards = [Card(suit, rank) for suit in self.suits for rank in self.ranks]
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.all_cards)

    def deal_one(self):
        if not self.all_cards:
            raise ValueError("Deck is empty")
        return self.all_cards.pop()
