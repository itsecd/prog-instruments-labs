import curses
import time
from game import GameBoard, Direction

def draw_board(window, board: GameBoard) -> None:
    window.clear()
    window.addstr(0, 0, str(board))
    window.refresh()

def handle_input(ch: int, board: GameBoard) -> None:
    mapping = {
        curses.KEY_UP: Direction.UP,
        curses.KEY_DOWN: Direction.DOWN,
        curses.KEY_LEFT: Direction.LEFT,
        curses.KEY_RIGHT: Direction.RIGHT
    }
    new_dir = mapping.get(ch)
    if new_dir and abs(board.direction.value - new_dir.value) != 2:
        board.direction = new_dir

def play(window, board: GameBoard) -> None:
    window.nodelay(True)
    while not board.finished:
        draw_board(window, board)
        time.sleep(0.1)
        board.move_snake()
        board.maybe_spawn_food()
        key = window.getch()
        if key != -1:
            handle_input(key, board)

def show_game_over(window, board: GameBoard) -> bool:
    msg = f"Game Over! Score: {len(board.snake) - 2}\nPress 'n' to restart or any key to quit."
    window.clear()
    window.addstr(0, 0, msg)
    window.refresh()
    key = window.getch()
    return chr(key).lower() == "n"