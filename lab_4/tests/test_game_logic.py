import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from game_logic import (  # noqa: E402
    can_place_ship,
    check_winner,
    coordinates_to_index,
    count_bulls_and_cows,
    index_to_coordinates,
    normalize_answer,
    slot_payout,
)

GRID = 10


def empty_board(size=GRID):
    return [[' '] * size for _ in range(size)]


class NormalizeAnswerTest(unittest.TestCase):
    def test_strips_spaces_and_lowercases(self):
        self.assertEqual(normalize_answer("  New York "), "newyork")

    def test_words_become_digits(self):
        self.assertEqual(normalize_answer("Two"), "2")
        self.assertEqual(normalize_answer("one hundred"), "1hundred")

    def test_digits_are_kept(self):
        self.assertEqual(normalize_answer("42"), "42")


class BullsAndCowsTest(unittest.TestCase):
    def test_exact_match(self):
        self.assertEqual(count_bulls_and_cows("1234", "1234"), (4, 0))

    def test_all_cows(self):
        self.assertEqual(count_bulls_and_cows("4321", "1234"), (0, 4))

    def test_mixed(self):
        self.assertEqual(count_bulls_and_cows("1243", "1234"), (2, 2))

    def test_no_common_digits(self):
        self.assertEqual(count_bulls_and_cows("5678", "1234"), (0, 0))


class SlotPayoutTest(unittest.TestCase):
    def test_three_of_a_kind(self):
        self.assertEqual(slot_payout(["A", "A", "A"], 10), (100, "jackpot"))

    def test_pair_in_any_position(self):
        for reels in (["A", "A", "B"], ["A", "B", "B"], ["A", "B", "A"]):
            with self.subTest(reels=reels):
                self.assertEqual(slot_payout(reels, 10), (30, "pair"))

    def test_no_match(self):
        self.assertEqual(slot_payout(["A", "B", "C"], 10), (0, "none"))


class CheckWinnerTest(unittest.TestCase):
    def test_empty_board_has_no_winner(self):
        self.assertIsNone(check_winner(empty_board(3)))

    def test_row(self):
        board = [["X", "X", "X"], [" ", "O", "O"], [" ", " ", " "]]
        self.assertEqual(check_winner(board), "X")

    def test_column(self):
        board = [["O", "X", " "], ["O", "X", " "], ["O", " ", "X"]]
        self.assertEqual(check_winner(board), "O")

    def test_both_diagonals(self):
        main = [["X", "O", " "], ["O", "X", " "], [" ", " ", "X"]]
        anti = [[" ", "O", "X"], ["O", "X", " "], ["X", " ", " "]]
        self.assertEqual(check_winner(main), "X")
        self.assertEqual(check_winner(anti), "X")

    def test_draw(self):
        board = [["X", "O", "X"], ["X", "O", "O"], ["O", "X", "X"]]
        self.assertIsNone(check_winner(board))


class CanPlaceShipTest(unittest.TestCase):
    def test_fits_on_empty_board(self):
        board = empty_board()
        self.assertTrue(can_place_ship(board, 0, 0, 3, 'H', GRID))
        self.assertTrue(can_place_ship(board, 0, 0, 3, 'V', GRID))

    def test_out_of_bounds(self):
        board = empty_board()
        self.assertFalse(can_place_ship(board, 0, 8, 3, 'H', GRID))
        self.assertFalse(can_place_ship(board, 8, 0, 3, 'V', GRID))

    def test_last_cell_is_allowed(self):
        self.assertTrue(can_place_ship(empty_board(), 0, 7, 3, 'H', GRID))

    def test_overlap(self):
        board = empty_board()
        board[0][1] = 'S'
        self.assertFalse(can_place_ship(board, 0, 0, 3, 'H', GRID))
        self.assertTrue(can_place_ship(board, 1, 0, 3, 'H', GRID))


class CoordinatesTest(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(coordinates_to_index("A", "1", GRID), (0, 0))
        self.assertEqual(coordinates_to_index("j", "10", GRID), (9, 9))

    def test_invalid(self):
        for letter, number in (("K", "1"), ("A", "0"), ("A", "11"), ("A", "x"), ("", "1"), (None, "1")):
            with self.subTest(letter=letter, number=number):
                self.assertEqual(coordinates_to_index(letter, number, GRID), (None, None))

    def test_round_trip(self):
        self.assertEqual(index_to_coordinates(4, 1), "B 5")
        row, col = coordinates_to_index("B", "5", GRID)
        self.assertEqual(index_to_coordinates(row, col), "B 5")


if __name__ == "__main__":
    unittest.main()