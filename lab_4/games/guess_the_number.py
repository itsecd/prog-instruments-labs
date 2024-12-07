import logging
import random


# Получаем существующий логгер
logger = logging.getLogger(__name__)


class GuessTheNumber:
    def __init__(self) -> None:
        """Инициализация игры 'Угадай число' с загадкой числа."""
        self.number_to_guess: int = random.randint(-100, 100)  # Изменено на диапазон от -100 до 100
        self.attempts: int = 0
        logger.info("Игра 'Угадай число' начата. Загаданное число: %d", self.number_to_guess)

    def play(self) -> None:
        """Основной игровой процесс."""
        print("Добро пожаловать в игру 'Угадай число'!")

        while True:
            guess = input(
                "Введите число от -100 до 100 (или 'выход' для завершения игры): "
            )
            if guess.lower() == "выход":
                print("Вы вышли из игры.")
                logger.info("Игрок вышел из игры.")
                break

            self.attempts += 1
            try:
                guess = int(guess)
            except ValueError:
                print("Пожалуйста, введите корректное число.")
                logger.warning("Пользователь ввел некорректное значение: %s", guess)
                continue

            self.check_guess(guess)

    def check_guess(self, guess: int) -> None:
        """
        Проверка угаданного числа и вывод подсказок.

        Args:
            guess (int): Угаданное число игроком.
        """
        difference: int = abs(self.number_to_guess - guess)
        logger.info("Игрок сделал попытку: %d", guess)

        if difference == 0:
            print(
                f"Поздравляем! Вы угадали число {self.number_to_guess} за {self.attempts} попыток."
            )
            logger.info("Игрок угадал число: %d за %d попыток.", self.number_to_guess, self.attempts)
        elif difference <= 5:
            print("Очень горячо! Вы очень близки к правильному числу.")
        elif difference <= 10:
            print("Горячо! Вы близки к правильному числу.")
        elif difference <= 20:
            print("Тепло. Вы находитесь в пределах 20.")
        elif guess < self.number_to_guess:
            print("Слишком маленькое число. Попробуйте снова.")
            logger.info("Игрок сделал слишком маленькую попытку: %d", guess)
        elif guess > self.number_to_guess:
            print("Слишком большое число. Попробуйте снова.")
            logger.info("Игрок сделал слишком большую попытку: %d", guess)
        else:
            print("Вы далеко от правильного ответа. Попробуйте снова.")
            logger.info("Игрок дал ответ, который далеко от правильного: %d", guess)


def guess_the_number() -> None:
    """Запуск игры 'Угадай число'."""
    game = GuessTheNumber()
    game.play()
