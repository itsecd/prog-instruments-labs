from BlackJackCLI import BlackJackCLI
from log_module import module_logger, log_errors
from loguru import logger


@log_errors(logger)
def main():
    main_logger = module_logger.bind(service="main")
    game = BlackJackCLI()
    main_logger.info("Start Game")
    game.play()
    main_logger.info("Finish game")



if __name__ == "__main__":
    main()

