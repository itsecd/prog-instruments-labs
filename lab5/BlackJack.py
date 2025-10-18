from Deck import Deck
from log_module import module_logger, log_errors
from Player import Player, Dealer

from loguru import logger


class BlackJack:
    """
    Класс с логикой игры БлэкДжек
    """
    def __init__(self):
        self.player = Player("Player", 100)
        self.dealer = Dealer("Dealer", 10000)
        self.game_num = 0
        self.DEALER_OPTIMAL_SUM = 17
        self.MAX_SUM = 21
        self.BlackJack_logger = module_logger.bind(service="BlackJack")

    @log_errors(logger)
    def start_round(self, bet: int):
        """
        Метод для запуска раунда
        :param bet: ставка
        """
        self.BlackJack_logger.info("Start new round")

        self.game_num += 1
        self.player.reset_hand()
        self.dealer.reset_hand()
        self.deck = Deck()
        self.deck.shuffle()

        bet_amount = self.player.make_bet(bet)

        for _ in range(2):
            self.player.add_card(self.deck.deal_one())
            self.dealer.add_card(self.deck.deal_one())

        return bet_amount

    @log_errors(logger)
    def player_hit(self):
        """
        Метод выдает карту игроку
        """
        card = self.deck.deal_one()
        self.player.add_card(card)
        return card

    @log_errors(logger)
    def dealer_play(self):
        """
        Метод для передачи хода диллеру
        """
        while self.dealer.should_hit() < self.DEALER_OPTIMAL_SUM:
            self.dealer.add_card(self.deck.deal_one())

    @log_errors(logger)
    def check_winner(self, bet: int):
        """
        Метод для проверки победителя
        :param bet: ставка
        """
        player_sum = self.player.hand_value()
        dealer_sum = self.dealer.hand_value()

        self.BlackJack_logger.info("Round finished")

        if player_sum > self.MAX_SUM:
            self.BlackJack_logger.info("Dealer win")
            return "dealer", 0

        if dealer_sum > self.MAX_SUM:
            self.BlackJack_logger.info("Player win")

            self.player.win(bet * 2)
            return "player", bet * 2

        if player_sum == self.MAX_SUM:
            self.BlackJack_logger.info("Player win")

            self.player.win(bet * 3)
            return "blackjack", bet * 3

        if player_sum > dealer_sum:
            self.BlackJack_logger.info("Player win")

            self.player.win(bet * 2)
            return "player", bet * 2
        elif player_sum < dealer_sum:
            self.BlackJack_logger.info("Dealer win")
            return "dealer", 0
        else:
            self.BlackJack_logger.info("Tie")

            self.player.win(bet)
            return "tie", bet
