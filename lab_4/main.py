import curses
import sys
from game import GameBoard
from ui import play, show_game_over

def main(window):
    rows, cols = 35, 70
    if len(sys.argv) >= 3:
        rows, cols = int(sys.argv[1]), int(sys.argv[2])

    board = GameBoard(rows, cols)
    while True:
        play(window, board)
        if not show_game_over(window, board):
            break
        board = GameBoard(rows, cols)

if __name__ == "__main__":
    curses.wrapper(main)