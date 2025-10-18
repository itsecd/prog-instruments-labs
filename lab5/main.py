from BlackJackCLI import BlackJackCLI
from logger import module_logger


def main():
    main_logger = module_logger.bind(service="main")
    game = BlackJackCLI()
    main_logger.info("Start Game")
    game.play()
    main_logger.info("Finish game")


if __name__ == "__main__":
    main()

