from Deck import Card
from log_module import module_logger, log_errors
from loguru import logger

class Player:
    """
    Класс игрока
    """

    @log_errors(logger)
    def __init__(self, name, chips):
        self.name = name
        self.chips = chips
        self.hand = []

    @log_errors(logger)
    def reset_hand(self):
        """
        Обнуление руки
        """
        self.hand = []

    @log_errors(logger)
    def add_card(self, card: Card):
        """
        Добавление карты в руку
        """
        self.hand.append(card)

    @log_errors(logger)
    def hand_value(self):
        """
        Сумма значений карт в руке
        """
        total_sum = sum(card.value() for card in self.hand)
        aces_count = sum(1 for card in self.hand if card.rank == "Ace")
        while total_sum > 21 and aces_count:
            total_sum -= 10
            aces_count -= 1
        return total_sum

    @log_errors(logger)
    def make_bet(self, bet: int):
        """
        Метод для создания ставки
        """
        if bet < 0:
            raise ValueError("Bet must be > 0")
        if bet > self.chips:
            raise ValueError("Not enough chips")
        self.chips -= bet
        return bet

    @log_errors(logger)
    def win(self, winnings: int):
        """
        Метод для начисления выигрыша
        """
        self.chips += winnings

    @log_errors(logger)
    def __str__(self):
        cards = ", ".join(str(card) for card in self.hand)
        return f"{self.name} hand: {cards}"


class Dealer(Player):
    """
    Класс дилера
    """

    @log_errors(logger)
    def should_hit(self):
        """
        Метод возвращает значение руки
        """
        return self.hand_value()
