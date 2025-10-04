import math
import pygame
import sys
import random


# global
SPEED = 0.36
SNAKE_SIZE = 9
APPLE_SIZE = SNAKE_SIZE
SEPARATION = 10
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 800
FPS = 25
KEY = {"UP": 1, "DOWN": 2, "LEFT": 3, "RIGHT": 4}
background_color = pygame.Color(0, 255, 255)
black = pygame.Color(0, 255, 255)


class Apple:
    def __init__(self, x, y, state):
        self.x = x
        self.y = y
        self.state = state
        self.color = pygame.color.Color("red")

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (self.x, self.y, APPLE_SIZE, APPLE_SIZE), 0)


class Segment:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.direction = KEY["UP"]
        self.color = "red"


class Snake:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.direction = KEY["UP"]
        self.stack = []

        self.stack.append(self)

        black_box = Segment(self.x, self.y + SEPARATION)
        black_box.direction = KEY["UP"]
        black_box.color = "NULL"
        self.stack.append(black_box)

    def move(self):
        last_element = len(self.stack) - 1
        while (last_element != 0):
            self.stack[last_element].direction = self.stack[last_element - 1].direction
            self.stack[last_element].x = self.stack[last_element - 1].x
            self.stack[last_element].y = self.stack[last_element - 1].y
            last_element -= 1
        if (len(self.stack) < 2):
            last_segment = self
        else:
            last_segment = self.stack.pop(last_element)
        last_segment.direction = self.stack[0].direction
        if (self.stack[0].direction == KEY["UP"]):
            last_segment.y = self.stack[0].y - (SPEED * FPS)
        elif (self.stack[0].direction == KEY["DOWN"]):
            last_segment.y = self.stack[0].y + (SPEED * FPS)
        elif (self.stack[0].direction == KEY["LEFT"]):
            last_segment.x = self.stack[0].x - (SPEED * FPS)
        elif (self.stack[0].direction == KEY["RIGHT"]):
            last_segment.x = self.stack[0].x + (SPEED * FPS)
        self.stack.insert(0, last_segment)

    def get_head(self):
        return (self.stack[0])

    def grow(self):
        last_element = len(self.stack) - 1
        self.stack[last_element].direction = self.stack[last_element].direction
        if (self.stack[last_element].direction == KEY["UP"]):
            new_segment = Segment(self.stack[last_element].x, self.stack[last_element].y - SNAKE_SIZE)
            black_box = Segment(new_segment.x, new_segment.y - SEPARATION)

        elif (self.stack[last_element].direction == KEY["DOWN"]):
            new_segment = Segment(self.stack[last_element].x, self.stack[last_element].y + SNAKE_SIZE)
            black_box = Segment(new_segment.x, new_segment.y + SEPARATION)

        elif (self.stack[last_element].direction == KEY["LEFT"]):
            new_segment = Segment(self.stack[last_element].x - SNAKE_SIZE, self.stack[last_element].y)
            black_box = Segment(new_segment.x - SEPARATION, new_segment.y)

        elif (self.stack[last_element].direction == KEY["RIGHT"]):
            new_segment = Segment(self.stack[last_element].x + SNAKE_SIZE, self.stack[last_element].y)
            black_box = Segment(new_segment.x + SEPARATION, new_segment.y)

        black_box.color = "NULL"
        self.stack.append(new_segment)
        self.stack.append(black_box)

    def iterate_segments(self, delta):
        pass

    def set_direction(self, direction):
        if (self.direction == KEY["RIGHT"] and direction == KEY["LEFT"] or self.direction == KEY[
            "LEFT"] and direction == KEY["RIGHT"]):
            pass
        elif (self.direction == KEY["UP"] and direction == KEY["DOWN"] or self.direction == KEY["DOWN"] and direction ==
              KEY["UP"]):
            pass
        else:
            self.direction = direction

    def get_rect(self):
        rect = (self.x, self.y)
        return rect

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def set_x(self, x):
        self.x = x

    def set_y(self, y):
        self.y = y

    def check_crash(self):
        counter = 1
        while (counter < len(self.stack) - 1):
            if (check_collision(self.stack[0], SNAKE_SIZE, self.stack[counter], SNAKE_SIZE) and self.stack[
                counter].color != "NULL"):
                return True
            counter += 1
        return False

    def draw(self, screen):
        pygame.draw.rect(screen, pygame.color.Color("yellow"),
                         (self.stack[0].x, self.stack[0].y, SNAKE_SIZE, SNAKE_SIZE), 0)
        counter = 1
        while (counter < len(self.stack)):
            if (self.stack[counter].color == "NULL"):
                counter += 1
                continue
            pygame.draw.rect(screen, pygame.color.Color("white"),
                             (self.stack[counter].x, self.stack[counter].y, SNAKE_SIZE, SNAKE_SIZE), 0)
            counter += 1


def get_key():
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                return KEY["UP"]
            elif event.key == pygame.K_DOWN:
                return KEY["DOWN"]
            elif event.key == pygame.K_LEFT:
                return KEY["LEFT"]
            elif event.key == pygame.K_RIGHT:
                return KEY["RIGHT"]
            elif event.key == pygame.K_ESCAPE:
                return "exit"
            elif event.key == pygame.K_y:
                return "yes"
            elif event.key == pygame.K_n:
                return "no"
        if event.type == pygame.QUIT:
            sys.exit()


def respawn_apple(apples, index, sx, sy):
    radius = math.sqrt((SCREEN_WIDTH / 2 * SCREEN_WIDTH / 2 + SCREEN_HEIGHT / 2 * SCREEN_HEIGHT / 2)) / 2
    angle = 999
    while (angle > radius):
        angle = random.uniform(0, 800) * math.pi * 2
        x = SCREEN_WIDTH / 2 + radius * math.cos(angle)
        y = SCREEN_HEIGHT / 2 + radius * math.sin(angle)
        if (x == sx and y == sy):
            continue
    new_apple = Apple(x, y, 1)
    apples[index] = new_apple


def respawn_apples(apples, quantity, sx, sy):
    counter = 0
    del apples[:]
    radius = math.sqrt((SCREEN_WIDTH / 2 * SCREEN_WIDTH / 2 + SCREEN_HEIGHT / 2 * SCREEN_HEIGHT / 2)) / 2
    angle = 999
    while (counter < quantity):
        while (angle > radius):
            angle = random.uniform(0, 800) * math.pi * 2
            x = SCREEN_WIDTH / 2 + radius * math.cos(angle)
            y = SCREEN_HEIGHT / 2 + radius * math.sin(angle)
            if ((x - APPLE_SIZE == sx or x + APPLE_SIZE == sx) and (
                    y - APPLE_SIZE == sy or y + APPLE_SIZE == sy) or radius - angle <= 10):
                continue
        apples.append(Apple(x, y, 1))
        angle = 999
        counter += 1


def check_collision(pos_a, size_a, pos_b, size_b):
    # size_a size of a | size_b size of B
    if (pos_a.x < pos_b.x + size_b and pos_a.x + size_a > pos_b.x and pos_a.y < pos_b.y + size_b and pos_a.y + size_a > pos_b.y):
        return True
    return False

def check_limits(entity):
    if (entity.x > SCREEN_WIDTH):
        entity.x = SNAKE_SIZE
    if (entity.x < 0):
        entity.x = SCREEN_WIDTH - SNAKE_SIZE
    if (entity.y > SCREEN_HEIGHT):
        entity.y = SNAKE_SIZE
    if (entity.y < 0):
        entity.y = SCREEN_HEIGHT - SNAKE_SIZE

def end_game():
    message = game_over_font.render("Game Over", 1, pygame.Color("white"))
    message_play_again = play_again_font.render("Play Again? Y/N", 1, pygame.Color("green"))
    screen.blit(message, (320, 240))
    screen.blit(message_play_again, (320 + 12, 240 + 40))

    pygame.display.flip()
    pygame.display.update()

    my_key = get_key()
    while (my_key != "exit"):
        if (my_key == "yes"):
            main()
        elif (my_key == "no"):
            break
        my_key = get_key()
        game_clock.tick(FPS)
    sys.exit()


def draw_score(score):
    score_numb = score_numb_font.render(str(score), 1, pygame.Color("red"))
    screen.blit(score_msg, (SCREEN_WIDTH - score_msg_size[0] - 60, 10))
    screen.blit(score_numb, (SCREEN_WIDTH - 45, 14))


def draw_game_time(gameTime):
    game_time = score_font.render("Time:", 1, pygame.Color("red"))
    game_time_numb = score_numb_font.render(str(gameTime / 1000), 1, pygame.Color("red"))
    screen.blit(game_time, (30, 10))
    screen.blit(game_time_numb, (105, 14))


def exit_screen():
    pass


def main():
    pygame.init()
    pygame.display.set_caption("$nAke bRo color fUll--FASAL ")
    pygame.font.init()
    random.seed()

    global screen, game_clock, score_font, score_numb_font
    global game_over_font, play_again_font, score_msg, score_msg_size

    # Screen initialization
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.HWSURFACE)

    # Resources
    score_font = pygame.font.Font(None, 38)
    score_numb_font = pygame.font.Font(None, 28)
    game_over_font = pygame.font.Font(None, 46)
    play_again_font = score_numb_font
    score_msg = score_font.render("Score:", 1, pygame.Color("red"))
    score_msg_size = score_font.size("Score")

    # Clock
    game_clock = pygame.time.Clock()

    # Game variables
    score = 0

    # Snake initialization
    my_snake = Snake(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)
    my_snake.set_direction(KEY["UP"])
    my_snake.move()
    start_segments = 3
    while (start_segments > 0):
        my_snake.grow()
        my_snake.move()
        start_segments -= 1

    # Apples
    max_apples = 1
    eaten_apple = False
    apples = [Apple(random.randint(60, SCREEN_WIDTH), random.randint(60, SCREEN_HEIGHT), 1)]
    respawn_apples(apples, max_apples, my_snake.x, my_snake.y)

    start_time = pygame.time.get_ticks()
    endgame = 0

    while (endgame != 1):
        game_clock.tick(FPS)

        # Input
        key_press = get_key()
        if key_press == "exit":
            endgame = 1

        # Collision check
        check_limits(my_snake)
        if (my_snake.check_crash() == True):
            end_game()

        for my_apple in apples:
            if (my_apple.state == 1):
                if (check_collision(my_snake.get_head(), SNAKE_SIZE, my_apple, APPLE_SIZE) == True):
                    my_snake.grow()
                    my_apple.state = 0
                    score += 5
                    eaten_apple = True

        # Position Update
        if (key_press):
            my_snake.set_direction(key_press)
        my_snake.move()

        # Respawning apples
        if (eaten_apple == True):
            eaten_apple = False
            respawn_apple(apples, 0, my_snake.get_head().x, my_snake.get_head().y)

        # Drawing
        screen.fill(background_color)
        for my_apple in apples:
            if (my_apple.state == 1):
                my_apple.draw(screen)

        my_snake.draw(screen)
        draw_score(score)
        game_time = pygame.time.get_ticks() - start_time
        draw_game_time(game_time)

        pygame.display.flip()
        pygame.display.update()


if __name__ == "__main__":
    main()