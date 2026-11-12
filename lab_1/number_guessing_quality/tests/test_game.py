import unittest
from unittest.mock import patch

from game import compare_guess, main


class CompareGuessTests(unittest.TestCase):
    def test_correct(self) -> None:
        self.assertEqual(compare_guess(42, 42), "correct")

    def test_above(self) -> None:
        self.assertEqual(compare_guess(42, 80), "above")

    def test_below(self) -> None:
        self.assertEqual(compare_guess(42, 8), "below")

    def test_invalid_maximum(self) -> None:
        with patch("builtins.input", return_value="nope"):
            with patch("builtins.print") as print_mock:
                main()
        print_mock.assert_called_once_with("Type a positive number next time.")

    def test_win(self) -> None:
        with patch("builtins.input", side_effect=["100", "42"]):
            with patch("game.random.randint", return_value=42):
                with patch("builtins.print") as print_mock:
                    main()
        self.assertEqual(print_mock.call_count, 2)
        print_mock.assert_any_call("You got it!")


if __name__ == "__main__":
    unittest.main()
