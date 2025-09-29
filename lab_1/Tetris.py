#=========================================================================
# Pygame based Tetris v0.1
#
# Copyright 2018 by Daehyuk Ahn
#
# Released under GPL
#=========================================================================
import sys
import time
import random
import pygame
import math

# define Helper class

class Rect:
    def __init__(self, x, y, w, h):
        self.left = x
        self.top = y
        self.right = x + w
        self.bottom = y + h

    def contains(self, x, y):
        # Return true if a point is inside the rectangle.
        return (self.left <= x <= self.right and
                self.top <= y <= self.bottom)

#
# Init global variables
#
TETRIS_SIZE = 24
TETRIS_WIDTH = 10
TETRIS_HEIGHT = 24

# Board Map area [TETRIS_WIDTH][TETRIS_HEIGHT] Max
tetris_board = [[0] * TETRIS_HEIGHT for i in range(TETRIS_WIDTH)]

# Play Surface
init_status = pygame.init()
pygame.display.set_caption("Tetris")

TETRIS_WINDOW = width, height = (
    TETRIS_WIDTH * TETRIS_SIZE * 2,
    TETRIS_HEIGHT * TETRIS_SIZE + 64,
)
tetris_screen = pygame.display.set_mode(TETRIS_WINDOW)

# Define Color
RED = pygame.Color(255, 0, 0)
BLUE = pygame.Color(0, 0, 255)
CYAN = pygame.Color(0, 255, 255)
BLACK = pygame.Color(0, 0, 0)
GRAY = pygame.Color(211, 211, 211)
WHITE = pygame.Color(255, 255, 255)
DARKGRAY = pygame.Color(128, 128, 128)

shape_char = ["I", "J", "L", "O", "S", "T", "Z"]
shape_colors = [
    (0,   255, 255), # I, Cyan
    (0,   0,   255), # J, Blue
    (255, 165, 0  ), # L, Orange
    (255, 255, 0  ), # O, Yellow
    (0,   255, 0  ), # S, Green
    (255, 0,   255), # T, Purple
    (255, 0,   0  ), # Z, Red
]
shape_angle = [0, 90, 180, 270]
shape_block = [
    [  # I
        [[0, 0], [1, 0], [2, 0], [3, 0]],
        [[2, 0], [2, 1], [2, 2], [2, 3]],
        [[0, 0], [1, 0], [2, 0], [3, 0]],
        [[1, 0], [1, 1], [1, 2], [1, 3]]
    ],
    [  # J
        [[0, 0], [1, 0], [2, 0], [2, 1]],
        [[1, 0], [1, 1], [1, 2], [0, 2]],
        [[0, 0], [0, 1], [1, 1], [2, 1]],
        [[1, 0], [2, 0], [1, 1], [1, 2]]
    ],
    [  # L
        [[0, 0], [1, 0], [2, 0], [0, 1]],
        [[0, 0], [1, 0], [1, 1], [1, 2]],
        [[0, 1], [1, 1], [2, 1], [2, 0]],
        [[1, 0], [1, 1], [1, 2], [2, 2]]
    ],
    [  # O
        [[0, 0], [1, 0], [0, 1], [1, 1]],
        [[0, 0], [1, 0], [0, 1], [1, 1]],
        [[0, 0], [1, 0], [0, 1], [1, 1]],
        [[0, 0], [1, 0], [0, 1], [1, 1]]
    ],
    [  # S
        [[1, 0], [2, 0], [0, 1], [1, 1]],
        [[1, 0], [1, 1], [2, 1], [2, 2]],
        [[1, 0], [2, 0], [0, 1], [1, 1]],
        [[0, 0], [0, 1], [1, 1], [1, 2]]
    ],
    [  # T
        [[0, 0], [1, 0], [2, 0], [1, 1]],
        [[0, 1], [1, 0], [1, 1], [1, 2]],
        [[1, 0], [0, 1], [1, 1], [2, 1]],
        [[1, 0], [1, 1], [1, 2], [2, 1]]
    ],
    [  # Z
        [[0, 0], [1, 0], [1, 1], [2, 1]],
        [[0, 1], [0, 2], [1, 0], [1, 1]],
        [[0, 0], [1, 0], [1, 1], [2, 1]],
        [[2, 0], [2, 1], [1, 1], [1, 2]]
    ]
]

shape_config = [[0, 0, 0, 0] * 4 for i in range(7)] # [7][4][0,1,2,3]

# Draw Screen
tetris_screen.fill(BLACK)

pygame.display.flip()


def make_shape_config():
    for s in range(len(shape_block)): # 7
        for a in range(len(shape_block[s])): # 4
            f, w, h = 3, 0, 0
            for i in range(len(shape_block[s][a])): # 4
                x, y = shape_block[s][a][i]
                if f > x: f = x
                if w < x: w = x
                if h < y: h = y
                # print("[{}, {}],".format(x, y), end="")
            w = w + 1 - f
            h = h + 1
            shape_config[s][a] = [f, w, h]
            # print(" = ", shape_config[s][a])
    return


def draw_tetris_board():
    for y in range(TETRIS_HEIGHT + 1):
        px = 16 + TETRIS_SIZE * TETRIS_WIDTH
        py = 16 + TETRIS_SIZE * y
        pygame.draw.line(tetris_screen, DARKGRAY, [16, py], [px, py], 1)

    for x in range(TETRIS_WIDTH + 1):
        px = 16 + TETRIS_SIZE * x
        py = 16 + TETRIS_SIZE * TETRIS_HEIGHT
        pygame.draw.line(tetris_screen, DARKGRAY, [px, 16], [px, py], 1)

    for y in range(TETRIS_HEIGHT):
        for x in range(TETRIS_WIDTH):
            s = tetris_board[x][y]
            if s >= 0:
                draw_tetris_block(x, y, shape_colors[s])
    return


def draw_tetris_block(x, y, c):
    # Check range is valid
    if (-1 < x < TETRIS_WIDTH) and (-1 < y < TETRIS_HEIGHT):
        px = 17 + TETRIS_SIZE * x
        py = 17 + TETRIS_SIZE * y
        pygame.draw.rect(tetris_screen, c, [px, py, 23, 23], 0)

    return


def draw_tetris_next(x, y, c):
    # Check range is valid
    if (-1 < x < 20) and (-1 < y < TETRIS_HEIGHT):
        px = 17 + TETRIS_SIZE * x
        py = 17 + TETRIS_SIZE * y
        pygame.draw.rect(tetris_screen, c, [px, py, 23, 23], 0)

    return


def draw_tetris_outline(x, y, c):
    # Check range is valid
    if (-1 < x < TETRIS_WIDTH) and (-1 < y < TETRIS_HEIGHT):
        px = 17 + TETRIS_SIZE * x
        py = 17 + TETRIS_SIZE * y
        pygame.draw.rect(tetris_screen, c, [px, py, 23, 23], 1)

    return


def is_conflict(x, y):
    if x < 0 or x >= TETRIS_WIDTH: return -1
    if y < 0 or y >= TETRIS_HEIGHT: return -1
    return tetris_board[x][y]


def draw_tetris(x, y, shape, angle):
    global g_ymax
    global g_game

    draw_tetris_board()

    # select current brick
    b = shape_block[shape][angle]
    f, w, h = shape_config[shape][angle]

    # check fallen tetris conflict
    for i in range(len(b)):
        nx, ny = b[i]
        if is_conflict(x + nx, y + ny) != -1:
            disp_start()
            g_game = False
            return

    # drawing fallen tetris
    for i in range(len(b)):
        nx, ny = b[i]
        draw_tetris_block(x + nx, y + ny, shape_colors[shape])

    # drawing fallen tetris
    for by in range(y, TETRIS_HEIGHT - h + 1):
        conflict = False
        for i in range(len(b)):
            nx, ny = b[i]
            if is_conflict(x + nx, by + ny) != -1:
                conflict = True
                g_ymax = by - 1
                break
        if conflict: break
        g_ymax = by

    for i in range(len(b)):
        nx, ny = b[i]
        draw_tetris_outline(x + nx, g_ymax + ny, shape_colors[shape])

    # select preview brick
    b = shape_block[g_next][0]
    f, w, h = shape_config[g_next][0]

    # drawing preview tetris
    for i in range(len(b)):
        nx, ny = b[i]
        draw_tetris_next(13 + nx, 0 + ny, shape_colors[g_next])

    return


def disp_score():
    global g_score, g_lines, g_level

    px = 17 + TETRIS_SIZE * 11
    py = 17 + TETRIS_SIZE * 20
    pygame.draw.rect(tetris_screen, BLUE, [px, py, 24 * 7, 24 * 4], 1)

    font = pygame.font.Font(None, 30)
    text = font.render("Lines " + str(g_lines), True, WHITE)
    tetris_screen.blit(text, [px + 12, py + 28 * 0 + 12])
    text = font.render("Level " + str(g_level), True, WHITE)
    tetris_screen.blit(text, [px + 12, py + 28 * 1 + 12])
    text = font.render("Score " + str(g_score), True, WHITE)
    tetris_screen.blit(text, [px + 12, py + 28 * 2 + 12])

    return


def disp_start():
    global g_score, g_lines, g_level

    px = 17 + TETRIS_SIZE * 2
    py = 17 + TETRIS_SIZE * 10
    pygame.draw.rect(tetris_screen, BLUE, [px, py, 24 * 15, 24 * 3], 0)

    font = pygame.font.Font(None, 30)
    if not g_game:
        text = font.render("Game Ready!", True, WHITE)
    else:
        text = font.render("Game Over!", True, WHITE)

    tetris_screen.blit(text, [px + 12, py + 28 * 0 + 12])
    text = font.render("Press 'N' key to start new game!", True, WHITE)
    tetris_screen.blit(text, [px + 12, py + 28 * 1 + 12])

    return


def process_timer(event):
    global g_char, g_angle, g_next
    global g_xpos, g_ypos
    global g_ymax
    global g_game

    if not g_game: return

    g_ypos = g_ypos + 1

    f, w, h = shape_config[g_char][g_angle]

    if g_ypos >= g_ymax:
        add_tetris(g_xpos, g_ypos, g_char, g_angle)
        g_xpos, g_ypos, g_angle = 3, 0, 0
        g_char = g_next
        g_next = random.randint(0, len(shape_char) - 1)

    tetris_screen.fill(BLACK)
    draw_tetris(g_xpos, g_ypos, g_char, g_angle)
    pygame.display.flip()

    return


# remove one line from top to bottom
def remove_line(y):
    # pull down lines
    for by in range(y, 0, -1):
        for bx in range(0, TETRIS_WIDTH):
            tetris_board[bx][by] = tetris_board[bx][by - 1]
    # erase top line
    for bx in range(0, TETRIS_WIDTH):
        tetris_board[bx][0] = -1
    return


# Add fallen tetris into board
def add_tetris(x, y, shape, angle):
    global g_score, g_lines, g_level, g_time
    scores = [0, 40, 100, 300, 120]

    b = shape_block[shape][angle]

    for i in range(len(b)):
        nx, ny = b[i]
        tetris_board[x + nx][y + ny] = shape

    # check if line is full
    c_lines = g_lines
    for by in range(TETRIS_HEIGHT - 1, 0, -1):
        full = True
        for bx in range(0, TETRIS_WIDTH):
            if tetris_board[bx][by] == -1:
                full = False
                break
        # flash effect should be add
        if full:
            # print("Full ", by)
            remove_line(by)
            g_lines += 1
            # g_score += 10

    # Calc score
    c_lines = g_lines - c_lines
    g_score += scores[c_lines]

    # Calc level and drop timer
    cLevel = int(g_lines / 10) + 1
    if g_level < cLevel:
        g_level = cLevel
        # level 1 = 500ms, level 50 and above = 50ms
        g_time = g_level if g_level < 50 else 50
        g_time = int(math.cos(math.pi / 100.0 * g_time) * 450) + 50
        print("Level = {}, Timer = {}ms".format(g_level, g_time))
        pygame.time.set_timer(pygame.USEREVENT, g_time)

    return


# shape_char = ["I", "J", "L", "O", "S", "T", "Z"]
shape_char = ["T", "S", "Z", "J", "L", "I", "O"]
shape_angle = [0, 90, 180, 270]
g_char, g_angle = 0, 0
g_xpos, g_ypos, g_ymax = 3, 0, 0
g_score, g_lines, g_level, g_next = 0, 0, 0, 0
g_game = False
g_time = 0


def key_down(event):
    global g_char, g_angle
    global g_xpos, g_ypos
    global g_game

    if not g_game:
        if event.key == pygame.K_n:
            print("g_game is False, Space pressed!")
            g_game = True
            new_game()
        return

    if event.key == pygame.K_RETURN:
        if (g_char + 1) < len(shape_char):
            g_char += 1
        else:
            g_char = 0
    elif event.key == pygame.K_UP:
        if (g_angle + 1) < len(shape_angle):
            g_angle += 1
        else:
            g_angle = 0
        # Get new shape
        f, w, h = shape_config[g_char][g_angle]
        # Adjust left side
        if g_xpos < 0: g_xpos = 0
        # Adjust right side
        if g_xpos > (TETRIS_WIDTH - (w + f)):
            g_xpos = TETRIS_WIDTH - (w + f)

    elif event.key == pygame.K_DOWN:
        process_timer(event)

    elif event.key == pygame.K_SPACE:
        g_ypos = g_ymax - 1
        process_timer(event)

    elif event.key == pygame.K_LEFT:
        # shape = shape_block[g_char][g_angle]
        f, w, h = shape_config[g_char][g_angle]
        # print(f, w, h)
        if g_xpos > (-f): g_xpos -= 1

    elif event.key == pygame.K_RIGHT:
        f, w, h = shape_config[g_char][g_angle]
        if g_xpos < (TETRIS_WIDTH - (w + f)): g_xpos += 1


    tetris_screen.fill(BLACK)
    draw_tetris(g_xpos, g_ypos, g_char, g_angle)

    return


def new_game():
    global g_char, g_angle, g_next
    global g_score, g_lines, g_level
    global g_game, g_time

    print("newGame called!")

    # Clean up Tetris values
    g_score, g_lines, g_level = 0, 0, 1
    g_angle = 0

    if not g_game:
        g_char = random.randint(0, len(shape_char) - 1)
        g_next = random.randint(0, len(shape_char) - 1)

    # Clean up Tetris board
    for y in range(TETRIS_HEIGHT):
        for x in range(TETRIS_WIDTH):
            tetris_board[x][y] = -1

    draw_tetris(g_xpos, g_ypos, g_char, g_angle)
    disp_score()
    pygame.display.flip()

    g_time = g_level if g_level < 50 else 50
    g_time = int(math.cos(math.pi / 100.0 * g_time) * 450) + 50
    print("Level = {}, Timer = {}ms".format(g_level, g_time))
    pygame.time.set_timer(pygame.USEREVENT, g_time)

    return


#--------------------------------------------------------------------------
# Main program
#--------------------------------------------------------------------------
def main():
    global shape_char, shape_angle
    global g_char, g_angle
    global g_xpos, g_ypos

    make_shape_config()
    new_game()
    disp_start()

    # pygame.time.set_timer(pygame.USEREVENT, 500)

    # main game loop
    while True:

        # Get Keyboard
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                key_down(event)
                disp_score()
                pygame.display.flip()
            elif event.type == pygame.USEREVENT:
                process_timer(event)
                disp_score()
                pygame.display.flip()


# run code
if __name__ == '__main__':
    main()