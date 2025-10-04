from config import suits, values, ranks

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
    def __init__(self):
        self.all_cards = []
        for suit in suits:
            for rank in ranks:
                # This assumes the Card class has already been defined!
                self.all_cards.append(Card(suit, rank))

    def shuffle(self):
        random.shuffle(self.all_cards)

    def deal_one(self):
        # we remove one card from the list of all_cards
        return self.all_cards.pop()
