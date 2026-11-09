import random
from typing import Literal


def compare_guess(secret: int, guess: int) -> Literal["correct", "above", "below"]:
    if guess == secret:
        return "correct"
    if guess > secret:
        return "above"
    return "below"


def main() -> None:
    top_of_range = input("Type a number: ")
    if not top_of_range.isdigit() or int(top_of_range) <= 0:
        print("Type a positive number next time.")
        return

    random_number = random.randint(0, int(top_of_range))
    guesses = 0

    while True:
        guesses += 1
        user_guess = input("Make a guess: ")
        if not user_guess.isdigit():
            print("Type a number next time.")
            continue

        result = compare_guess(random_number, int(user_guess))
        if result == "correct":
            print("You got it!")
            break
        if result == "above":
            print("You were above the number.")
        else:
            print("You were below the number.")

    print("You got it in", guesses, "guesses.")


if __name__ == "__main__":
    main()
