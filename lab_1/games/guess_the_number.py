import random


class GuessTheNumber:
    def __init__(self):
        self.number_to_guess = random.randint(-100, 100)  # Изменено на диапазон от -100 до 100
        self.attempts = 0

    def play(self):
        print("Добро пожаловать в игру 'Угадай число'!")

        while True:
            guess = input(
                "Введите число от -100 до 100 (или 'выход' для завершения игры): "
            )
            if guess.lower() == "выход":
                print("Вы вышли из игры.")
                break

            self.attempts += 1
            try:
                guess = int(guess)
            except ValueError:
                print("Пожалуйста, введите корректное число.")
                continue

            self.check_guess(guess)

    def check_guess(self, guess):
        """Проверка угаданного числа и вывод подсказок."""
        difference = abs(self.number_to_guess - guess)

        if difference == 0:
            print(
                f"Поздравляем! Вы угадали число {self.number_to_guess} за {self.attempts} попыток."
            )
        elif difference <= 5:
            print("Очень горячо! Вы очень близки к правильному числу.")
        elif difference <= 10:
            print("Горячо! Вы близки к правильному числу.")
        elif difference <= 20:
            print("Тепло. Вы находитесь в пределах 20.")
        elif guess < self.number_to_guess:
            print("Слишком маленькое число. Попробуйте снова.")
        elif guess > self.number_to_guess:
            print("Слишком большое число. Попробуйте снова.")
        else:
            print("Вы далеко от правильного ответа. Попробуйте снова.")


def guess_the_number():
    game = GuessTheNumber()
    game.play()
