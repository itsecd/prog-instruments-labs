from BlackJack import BlackJack
from config import BLACKJACK_STR, START_MESSAGE
from log_module import module_logger, log_errors

from loguru import logger


class BlackJackCLI:
    """
    Класс реализующий логику игры БлэкДжек через ввод через командную строку
    """

    @log_errors(logger)
    def __init__(self):
        self.game = BlackJack()
        self.BlackJackCLI_logger = module_logger.bind(service="BlackJackCLI")

    @log_errors(logger)
    def play(self):
        """
        Метод запускающий игру
        """
        self.BlackJackCLI_logger.info("Game start")
        print(BLACKJACK_STR)
        print(START_MESSAGE)

        while self.game.player.chips > 0:
            try:
                bet = int(input("Enter the amount of chips you want to bet: "))
                bet_amount = self.game.start_round(bet)

                print(self.game.player)
                print(f"Dealer cards are {self.game.dealer.hand[0]} and Hidden")

                while self.game.player.hand_value() < 21:
                    move = input("Do you want to hit or stand?:").lower()
                    if move == "hit":
                        card = self.game.player_hit()
                        print(f"\nThe player hits card :{card}")
                    elif move == "stand":
                        break
                    else:
                        print("Invalid option")

                self.game.dealer_play()

                print("\nFinal Hands:")
                print(self.game.player)
                print(self.game.dealer)

                winner, winnings = self.game.check_winner(bet_amount)
                match winner:
                    case "player":
                        print(f"You win! +{winnings} chips")
                    case "blackjack":
                        print(f"Blackjack! You win {winnings} chips!")
                    case "tie":
                        print("It's a tie.")
                    case "dealer":
                        print("Dealer wins!")

                print(f"Chips left: {self.game.player.chips}\n")

                cont = input("Play another round? (y/n): ").lower()
                if cont != "y":
                    break

            except ValueError as e:
                print(f"Error: {e}")
                continue

        print("\nGame over! You finished with", self.game.player.chips, "chips.")
        self.BlackJackCLI_logger.info("Game finish")
