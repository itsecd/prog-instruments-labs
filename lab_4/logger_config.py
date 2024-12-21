import logging


def setup_logger(name):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()  # Вывод в консоль
            # logging.FileHandler('app.log') # Вывод в файл
        ]
    )
    logger = logging.getLogger(name)
    return logger
