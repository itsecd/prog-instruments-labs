"""Pure game logic: no Tkinter, so it can be unit-tested without a window."""

WORD_TO_DIGIT = {
    'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
    'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
}

EMPTY = ' '
SHIP = 'S'

# Slot machine payout multipliers
JACKPOT_MULTIPLIER = 10
PAIR_MULTIPLIER = 3


def empty_board(size):
    """Returns a size x size grid of empty cells."""
    return [[EMPTY for _ in range(size)] for _ in range(size)]


def normalize_answer(answer):
    """Normalizes answer by removing spaces and converting written numbers to digits."""
    normalized = answer.strip().lower().replace(" ", "")
    for word, digit in WORD_TO_DIGIT.items():
        normalized = normalized.replace(word, digit)
    return normalized


def count_bulls_and_cows(guess, secret):
    """Returns (bulls, cows) for two strings of unique digits.

    Bulls are correct digits in the correct position, cows are correct
    digits in the wrong position.
    """
    bulls = sum(g == s for g, s in zip(guess, secret))
    cows = sum(g in secret for g in guess) - bulls
    return bulls, cows


def slot_payout(reels, bet):
    """Returns (winnings, outcome) for three reels.

    outcome is 'jackpot' (three equal), 'pair' (exactly two equal) or 'none'.
    """
    reel1, reel2, reel3 = reels
    if reel1 == reel2 == reel3:
        return bet * JACKPOT_MULTIPLIER, 'jackpot'
    if reel1 == reel2 or reel2 == reel3 or reel1 == reel3:
        return bet * PAIR_MULTIPLIER, 'pair'
    return 0, 'none'


def check_winner(board):
    """Returns 'X' or 'O' if that player has a full line on a 3x3 board, else None."""
    lines = []
    lines.extend(board[i] for i in range(3))
    lines.extend([board[0][j], board[1][j], board[2][j]] for j in range(3))
    lines.append([board[0][0], board[1][1], board[2][2]])
    lines.append([board[0][2], board[1][1], board[2][0]])
    for a, b, c in lines:
        if a == b == c != EMPTY:
            return a
    return None


def can_place_ship(board, row, col, size, direction, grid_size):
    """Checks that a ship fits on the board and does not overlap another ship.

    direction is 'H' (horizontal) or 'V' (vertical).
    """
    if direction == 'H':
        if col + size > grid_size:
            return False
        return all(board[row][col + i] != SHIP for i in range(size))
    if row + size > grid_size:
        return False
    return all(board[row + i][col] != SHIP for i in range(size))


def coordinates_to_index(letter, number, grid_size):
    """Converts e.g. ('B', '5') to (row, col) = (4, 1); (None, None) if invalid."""
    try:
        col = ord(letter.upper()) - ord('A')
        row = int(number) - 1
    except (ValueError, TypeError, AttributeError):
        return None, None
    if not (0 <= col < grid_size and 0 <= row < grid_size):
        return None, None
    return row, col


def index_to_coordinates(row, col):
    """Converts (row, col) = (4, 1) to the label 'B 5'."""
    return f"{chr(ord('A') + col)} {row + 1}"