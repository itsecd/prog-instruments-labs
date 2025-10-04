from Deck import Deck, Card


class Player:
    def __init__(self, name, chips):
        self.name = name
        self.chips = chips
        self.hand = []

    def reset_hand(self):
        self.hand = []

    def add_card(self, card: Card):
        self.hand.append(card)

    def hand_value(self):
        total_sum = sum(card.value() for card in self.hand)
        aces_count = sum(1 for card in self.hand if card.rank == "Ace")
        while total_sum > 21 and aces_count:
            total_sum -= 10
            aces_count -= 1
        return total_sum

    def make_bet(self, bet: int):
        if bet < 0:
            raise ValueError("Bet must be > 0")
        if bet > self.chips:
            raise ValueError("Not enough chips")
        self.chips -= bet
        return bet

    def win(self, winnings: int):
        self.chips += winnings

    def __str__(self):
        cards = ", ".join(str(card) for card in self.hand)
        return f"{self.name} hand: {cards}"


class Dealer(Player):

    def should_hit(self):
        return self.hand_value()
