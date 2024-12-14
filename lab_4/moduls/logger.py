import logging
import encodings

def create_logger(name='my_logger'):
    """Создает и возвращает логгер."""
    logger = logging.getLogger(name)
    if logger.hasHandlers():  # Проверяем, есть ли уже обработчики
        return logger  # Если есть, возвращаем существующий логгер

    logger.setLevel(logging.INFO)

    console = logging.StreamHandler()
    # Уровень логгера будет наследован обработчиком
    # console.setLevel(logging.INFO)  # Это уже не нужно

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    console.setFormatter(formatter)

    logger.addHandler(console)

    return logger


if __name__ == "__main__":
    logger = create_logger()
    logger.info("Запуск процедуры генерации и сериализации ключей")
    logger.warning("Возможны проблемы с генерацией ключей.")
    logger.error("Ошибка генерации ключей!")
