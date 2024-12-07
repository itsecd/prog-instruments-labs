import random
import logging
from typing import List


# Получаем существующий логгер
logger = logging.getLogger(__name__)


class Hangman:
    def __init__(self) -> None:
        """Инициализация игры в Виселицу с заданным списком слов."""
        self.words: List[str] = [
            "python",
            "программирование",
            "виселица",
            "игра",
            "разработка"
        ]
        self.word: str = random.choice(self.words)  # Загадать случайное слово
        self.word_completion: str = "_" * len(self.word)  # Скрытое слово
        self.guessed: bool = False  # Угадано ли слово
        self.guessed_letters: List[str] = []  # Угаданные буквы
        self.tries: int = 6  # Количество попыток
        logger.info("Игра в Виселицу начата. Загаданное слово: %s", self.word)

    def display_hangman(self) -> str:
        """Отображение состояния виселицы в зависимости от оставшихся попыток.

        Returns:
            str: Текущий этап виселицы.
        """
        stages: List[str] = [
            """
               ------
               |    |
               |    O
               |   /|\\
               |   / \\
               |
            """,
            """
               ------
               |    |
               |    O
               |   /|\\
               |   /
               |
            """,
            """
               ------
               |    |
               |    O
               |   /|
               |
               |
            """,
            """
               ------
               |    |
               |    O
               |    |
               |
               |
            """,
            """
               ------
               |    |
               |    O
               |
               |
               |
            """,
            """
               ------
               |    |
               |
               |
               |
               |
            """,
            """
               ------
               |
               |
               |
               |
               |
            """,
        ]
        return stages[self.tries]


    def play(self) -> None:
        """Основной игровой процесс."""
        print("Давайте играть в Виселицу!")
        print(self.display_hangman())
        print(self.word_completion)
        print("\n")

        while not self.guessed and self.tries > 0:
            guess: str = input("Угадайте букву: ").lower()

            if len(guess) != 1 or not guess.isalpha():
                print("Пожалуйста, введите одну букву.")
                logger.warning("Пользователь ввел недопустимый ввод: %s", guess)
                continue

            if guess in self.guessed_letters:
                print("Вы уже угадывали эту букву. Попробуйте другую.")
                logger.info("Пользователь уже угадывал букву: %s", guess)
                continue

            self.guessed_letters.append(guess)
            logger.info("Пользователь угадал букву: %s", guess)

            if guess in self.word:
                print("Хорошо! Буква есть в слове.")
                self.word_completion = "".join(
                    [
                        letter if letter in self.guessed_letters else "_"
                        for letter in self.word
                    ]
                )
            else:
                print("Увы, такой буквы нет в слове.")
                self.tries -= 1
                logger.info("Неверная буква: %s. Осталось попыток: %d", guess, self.tries)

            print(self.display_hangman())
            print(self.word_completion)
            print("\n")

            if "_" not in self.word_completion:
                self.guessed = True

        if self.guessed:
            print("Поздравляем! Вы угадали слово:", self.word)
            logger.info("Игрок угадал слово: %s", self.word)
        else:
            print("Вы проиграли. Загаданное слово было:", self.word)
            logger.info("Игрок проиграл. Загаданное слово: %s", self.word)


def hangman() -> None:
    """Запуск игры в Виселицу."""
    game = Hangman()
    game.play()
