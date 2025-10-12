import random

import pygame

pygame.font.init()

# Constant
S_WIDTH = 800
S_HEIGHT = 700
BLOCK_SIZE = 30
PLAY_WIDTH = 300  # mean 10 block
PLAY_HEIGHT = 600  # mean 20 block

TOP_LEFT_X = (S_WIDTH - PLAY_WIDTH) // 2
TOP_LEFT_Y = S_HEIGHT - PLAY_HEIGHT

# Shape Format
S = [['.....',
      '.....',
      '..00.',
      '.00..',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '...0.',
      '.....']]

Z = [['.....',
      '.....',
      '.00..',
      '..00.',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '.0...',
      '.....']]

I = [['..0..',
      '..0..',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '0000.',
      '.....',
      '.....',
      '.....']]

O = [['.....',
      '.....',
      '.00..',
      '.00..',
      '.....']]

J = [['.....',
      '.0...',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..00.',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '...0.',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '.00..',
      '.....']]

L = [['.....',
      '...0.',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '..00.',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '.0...',
      '.....'],
     ['.....',
      '.00..',
      '..0..',
      '..0..',
      '.....']]

T = [['.....',
      '..0..',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '..0..',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '..0..',
      '.....']]

shapes = [S, Z, I, O, J, L, T]
shape_colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 165, 0), (0, 0, 255), (128, 0, 128)]


class Piece:
    def __init__(self, x, y, shape):
        self.x = x
        self.y = y
        self.shape = shape
        self.color = shape_colors[shapes.index(shape)]
        self.rotation = 0


# locked_pos is position that other pieces already in the grid
def create_grid(locked_pos={}):
    grid = [[(0, 0, 0) for _ in range(10)] for _ in range(20)]  # also contain block color that paint the empty block in grid too
    for y in range(len(grid)):
        for x in range(len(grid[y])):
            if (x, y) in locked_pos:
                grid[y][x] = locked_pos[(x, y)]  # update color of block that other pieces already in
    return grid


def convert_shape_format(piece):
    positions = []
    format_shape = piece.shape[piece.rotation % len(piece.shape)]

    for i, line in enumerate(format_shape):
        row = list(line)
        for j, column in enumerate(row):
            if column == '0':
                positions.append((piece.x + j, piece.y + i))

    for i, pos in enumerate(positions):
        positions[i] = (pos[0] - 2, pos[1] - 4)

    return positions


def valid_space(piece, grid):
    # only accept if that position is empty (0, 0, 0)
    accepted_pos = [[(j, i) for j in range(10) if grid[i][j] == (0, 0, 0)] for i in range(20)]  # [[(0, 0), (1, 0), (2, 0)...], [(0, 1), (1, 1), (2, 1)...]...]
    accepted_pos = [j for sub in accepted_pos for j in sub]  # [(0, 0), (1, 0), (2, 0), ..., (0, 1), (1, 1), (2, 1), ...]

    formatted_shape = convert_shape_format(piece)

    for pos in formatted_shape:
        if pos not in accepted_pos:
            if pos[1] > -1:
                return False
    return True


def check_lost(positions):
    for pos in positions:
        x, y = pos
        if y < -1:
            return True

    return False


def get_shape():
    return Piece(5, 0, random.choice(shapes))


def draw_text_middle(surface, text, size, color):
    font = pygame.font.SysFont('comicsans', size, bold=True)
    label = font.render(text, True, color)

    surface.blit(label, (TOP_LEFT_X + PLAY_WIDTH / 2 - label.get_width() / 2, TOP_LEFT_Y + PLAY_HEIGHT / 2 - label.get_height() / 2))


def draw_grid(surface, grid):
    for i in range(len(grid)):
        # draw parallel lines with axis x
        pygame.draw.line(surface, (128, 128, 128), (TOP_LEFT_X, TOP_LEFT_Y + i * BLOCK_SIZE), (TOP_LEFT_X + PLAY_WIDTH, TOP_LEFT_Y + i * BLOCK_SIZE))
        for j in range(len(grid[i])):
            # draw parallel lines with axis y
            pygame.draw.line(surface, (128, 128, 128), (TOP_LEFT_X + j * BLOCK_SIZE, TOP_LEFT_Y), (TOP_LEFT_X + j * BLOCK_SIZE, TOP_LEFT_Y + PLAY_HEIGHT))


def clear_rows(grid, locked_positions):
    index = 0
    increment = 0
    for i in range(len(grid) - 1, -1, -1):
        row = grid[i]
        if (0, 0, 0) not in row:
            increment += 1
            index = i
            for j in range(len(row)):
                try:
                    del locked_positions[(j, i)]
                except:
                    continue

    if increment > 0:
        # sorted(list(locked_positions), key=lambda pos: pos[1]) sort locked_positions by y ascending
        # and [::-1] for reversing from below to above
        for key in sorted(list(locked_positions), key=lambda pos: pos[1])[::-1]:
            x, y = key
            if y < index:
                new_key = (x, y + increment)
                locked_positions[new_key] = locked_positions.pop(key)

    return increment


def draw_next_shape(piece, surface):
    font = pygame.font.SysFont('comicsans', 30)
    label = font.render('Next Shape', True, (255, 255, 255))
    sx = TOP_LEFT_X + PLAY_WIDTH + 50
    sy = TOP_LEFT_Y + PLAY_HEIGHT // 3

    format_shape = piece.shape[piece.rotation % len(piece.shape)]

    for i, line in enumerate(format_shape):
        row = list(line)
        for j, column in enumerate(row):
            if column == '0':
                pygame.draw.rect(surface, piece.color, (sx + j * BLOCK_SIZE, sy + i * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 0)

    surface.blit(label, (sx + 10, sy - 30))


def update_score(score):
    high_score = max_score()
    with open('scores.txt', 'w') as f:
        if int(high_score) < score:
            f.write(str(score))


def max_score():
    with open('scores.txt', 'r') as f:
        lines = f.readlines()
        high_score = lines[0].strip()

    return high_score


def draw_window(surface, grid, score=0, high_score=0):
    surface.fill((0, 0, 0))

    pygame.font.init()
    font = pygame.font.SysFont('comicsans', 60)
    label = font.render('Tetris', True, (255, 255, 255))

    # draw label in top center of play screen
    surface.blit(label, (TOP_LEFT_X + PLAY_WIDTH / 2 - (label.get_width() / 2), 30))

    # draw score
    font = pygame.font.SysFont('comicsans', 30)
    label = font.render('Score: ' + str(score), True, (255, 255, 255))
    sx = TOP_LEFT_X + PLAY_WIDTH + 50
    sy = TOP_LEFT_Y + PLAY_HEIGHT // 3
    surface.blit(label, (sx + 20, sy + 160))

    # draw high score
    label = font.render('High Score: ' + str(high_score), True, (255, 255, 255))
    surface.blit(label, (TOP_LEFT_X - 200, TOP_LEFT_Y))

    # draw play screen
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            pygame.draw.rect(surface, grid[i][j], (TOP_LEFT_X + j * BLOCK_SIZE, TOP_LEFT_Y + i * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 0)

    # draw border of play screen
    pygame.draw.rect(surface, (255, 0, 0), (TOP_LEFT_X, TOP_LEFT_Y, PLAY_WIDTH, PLAY_HEIGHT), 5)

    # draw grid
    draw_grid(surface, grid)


def main(surface):
    locked_positions = {}

    change_piece = False
    run = True
    is_quit = False
    current_piece = get_shape()
    next_piece = get_shape()
    clock = pygame.time.Clock()
    fall_time = 0
    fall_speed = 0.5
    level_time = 0
    score = 0
    high_score = int(max_score())

    while run:
        grid = create_grid(locked_positions)
        fall_time += clock.get_rawtime()
        level_time += clock.get_rawtime()
        clock.tick()

        if level_time / 1000 > 5:
            level_time = 0
            if fall_speed > 0.3:
                fall_speed -= 0.005

        if fall_time / 1000 > fall_speed:
            fall_time = 0
            current_piece.y += 1
            if not (valid_space(current_piece, grid)) and current_piece.y > 0:
                current_piece.y -= 1
                change_piece = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                # if we call pygame.display.quit() here, the surface which drawing function use down there will be dead display
                # and it will throw exception
                is_quit = True

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    current_piece.x -= 1
                    if not (valid_space(current_piece, grid)):
                        current_piece.x += 1
                if event.key == pygame.K_RIGHT:
                    current_piece.x += 1
                    if not (valid_space(current_piece, grid)):
                        current_piece.x -= 1
                if event.key == pygame.K_DOWN:
                    current_piece.y += 1
                    if not (valid_space(current_piece, grid)):
                        current_piece.y -= 1
                if event.key == pygame.K_UP:
                    current_piece.rotation += 1
                    if not (valid_space(current_piece, grid)):
                        current_piece.rotation -= 1

        shape_pos = convert_shape_format(current_piece)
        # put color of shape into grid
        for i in range(len(shape_pos)):
            x, y = shape_pos[i]
            if y > -1:
                grid[y][x] = current_piece.color

        if change_piece:
            for pos in shape_pos:
                p = (pos[0], pos[1])
                locked_positions[p] = current_piece.color
            current_piece = next_piece
            next_piece = get_shape()
            change_piece = False
            score += clear_rows(grid, locked_positions) * 10

        # window must be drawn first
        draw_window(surface, grid, score, high_score)
        draw_next_shape(next_piece, surface)
        pygame.display.update()

        if check_lost(locked_positions):
            draw_text_middle(surface, 'YOU LOST!', 80, (255, 255, 255))
            pygame.display.update()
            pygame.time.delay(1500)
            run = False
            update_score(score)

    # we couldn't call pygame.display.quit() here too because same thing happen in while loop of main_menu
    return is_quit


def main_menu(surface):
    run = True
    while run:
        surface.fill((0, 0, 0))
        draw_text_middle(surface, 'Press any key to play!', 60, (255, 255, 255))
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                run = not main(surface)
    pygame.display.quit()


window = pygame.display.set_mode((S_WIDTH, S_HEIGHT))
pygame.display.set_caption('Tetris')
main_menu(window)