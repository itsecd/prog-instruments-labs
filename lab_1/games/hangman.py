import random


class Hangman:
    def __init__(self):
        self.words = ["python", "программирование", "виселица", "игра", "разработка"]
        self.word = random.choice(self.words)  # Загадать случайное слово
        self.wordCompletion = "_" * len(self.word)  # Скрытое слово
        self.guessed = False  # Угадано ли слово
        self.guessedLetters = []  # Угаданные буквы
        self.tries = 6  # Количество попыток

    def displayHangman(self):
        """Отображение состояния виселицы в зависимости от оставшихся попыток."""
        stages = [
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

    def play(self):
        """Основной игровой процесс."""
        print("Давайте играть в Виселицу!")
        print(self.displayHangman())
        print(self.wordCompletion)
        print("\n")

        while not self.guessed and self.tries > 0:
            guess = input("Угадайте букву: ").lower()

            if len(guess) != 1 or not guess.isalpha():
                print("Пожалуйста, введите одну букву.")
                continue

            if guess in self.guessedLetters:
                print("Вы уже угадывали эту букву. Попробуйте другую.")
                continue

            self.guessedLetters.append(guess)

            if guess in self.word:
                print("Хорошо! Буква есть в слове.")
                self.wordCompletion = "".join(
                    [
                        letter if letter in self.guessedLetters else "_"
                        for letter in self.word
                    ]
                )
            else:
                print("Увы, такой буквы нет в слове.")
                self.tries -= 1

            print(self.displayHangman())
            print(self.wordCompletion)
            print("\n")

            if "_" not in self.wordCompletion:
                self.guessed = True

        if self.guessed:
            print("Поздравляем! Вы угадали слово:", self.word)
        else:
            print("Вы проиграли. Загаданное слово было:", self.word)


def hangman():
    game = Hangman()
    game.play()
