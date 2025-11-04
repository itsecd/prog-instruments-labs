import math
import pygame
import random
import sys


pygame.init()
pygame.display.set_caption("Simple Snake Game")
pygame.font.init()
random.seed()

SPEED = 0.36
SNAKE_SIZE = 9
APPLE_SIZE = SNAKE_SIZE
SEPARATION = 10
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 800
FPS = 25
KEY = {"UP": 1, "DOWN": 2, "LEFT": 3, "RIGHT": 4}

SCORE_FONT = pygame.font.Font(None, 38)
SCORE_NUMB_FONT = pygame.font.Font(None, 28)
GAME_OVER_FONT = pygame.font.Font(None, 46)
PLAY_AGAIN_FONT = SCORE_NUMB_FONT
SCORE_MSG = SCORE_FONT.render("Score:", 1, pygame.Color("yellow"))
SCORE_MSG_SIZE = SCORE_FONT.size("Score")
BACKGROUND_COLOR = pygame.Color(0, 0, 0)

SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.HWSURFACE)
GAME_CLOCK = pygame.time.Clock()


def check_collision(pos_a, a_s, pos_b, b_s):
    if pos_a.x < pos_b.x + b_s and pos_a.x + a_s > pos_b.x and \
            pos_a.y < pos_b.y + b_s and pos_a.y + a_s > pos_b.y:
        return True

    return False


def check_limits(snake):
    if snake.x > SCREEN_WIDTH:
        snake.x = SNAKE_SIZE

    if snake.x < 0:
        snake.x = SCREEN_WIDTH - SNAKE_SIZE

    if snake.y > SCREEN_HEIGHT:
        snake.y = SNAKE_SIZE

    if snake.y < 0:
        snake.y = SCREEN_HEIGHT - SNAKE_SIZE


class Apple:
    def __init__(self, x, y, state):
        self.x = x
        self.y = y
        self.state = state
        self.color = pygame.color.Color("green")

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (self.x, self.y, APPLE_SIZE, APPLE_SIZE), 0)


class Segment:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.direction = KEY["UP"]
        self.color = "white"


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
        while last_element != 0:
            self.stack[last_element].direction = self.stack[last_element - 1].direction
            self.stack[last_element].x = self.stack[last_element - 1].x
            self.stack[last_element].y = self.stack[last_element - 1].y
            last_element -= 1

        if len(self.stack) < 2:
            last_segment = self
        else:
            last_segment = self.stack.pop(last_element)
        last_segment.direction = self.stack[0].direction

        if self.stack[0].direction == KEY["UP"]:
            last_segment.y = self.stack[0].y - (SPEED * FPS)

        elif self.stack[0].direction == KEY["DOWN"]:
            last_segment.y = self.stack[0].y + (SPEED * FPS)

        elif self.stack[0].direction == KEY["LEFT"]:
            last_segment.x = self.stack[0].x - (SPEED * FPS)

        elif self.stack[0].direction == KEY["RIGHT"]:
            last_segment.x = self.stack[0].x + (SPEED * FPS)

        self.stack.insert(0, last_segment)

    def get_head(self):
        return self.stack[0]

    def grow(self):
        last_element = len(self.stack) - 1
        self.stack[last_element].direction = self.stack[last_element].direction
        new_segment = Segment(0, 0)
        black_box = Segment(0, 0)

        if self.stack[last_element].direction == KEY["UP"]:
            new_segment = Segment(self.stack[last_element].x, self.stack[last_element].y - SNAKE_SIZE)
            black_box = Segment(new_segment.x, new_segment.y - SEPARATION)

        elif self.stack[last_element].direction == KEY["DOWN"]:
            new_segment = Segment(self.stack[last_element].x, self.stack[last_element].y + SNAKE_SIZE)
            black_box = Segment(new_segment.x, new_segment.y + SEPARATION)

        elif self.stack[last_element].direction == KEY["LEFT"]:
            new_segment = Segment(self.stack[last_element].x - SNAKE_SIZE, self.stack[last_element].y)
            black_box = Segment(new_segment.x - SEPARATION, new_segment.y)

        elif self.stack[last_element].direction == KEY["RIGHT"]:
            new_segment = Segment(self.stack[last_element].x + SNAKE_SIZE, self.stack[last_element].y)
            black_box = Segment(new_segment.x + SEPARATION, new_segment.y)

        black_box.color = "NULL"
        self.stack.append(new_segment)
        self.stack.append(black_box)

    def set_direction(self, direction):
        if not ((self.direction == KEY["RIGHT"] and direction == KEY["LEFT"] or
                 self.direction == KEY["LEFT"] and direction == KEY["RIGHT"]) or
                (self.direction == KEY["UP"] and direction == KEY["DOWN"] or
                 self.direction == KEY["UP"] and direction == KEY["DOWN"])):
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

    def check_crashing(self):
        counter = 1
        while counter < len(self.stack) - 1:
            if check_collision(self.stack[0], SNAKE_SIZE, self.stack[counter], SNAKE_SIZE) and \
                    self.stack[counter].color != "NULL":
                return True

            counter += 1

        return False
    
    def draw(self, screen):
        pygame.draw.rect(screen, pygame.color.Color("green"),
                         (self.stack[0].x, self.stack[0].y, SNAKE_SIZE, SNAKE_SIZE), 0)
        counter = 1

        while counter < len(self.stack):
            if self.stack[counter].color == "NULL":
                counter += 1
                continue

            pygame.draw.rect(screen, pygame.color.Color("yellow"),
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
            sys.exit(0)


def end_game():
    message = GAME_OVER_FONT.render("Game Over", 1, pygame.Color("white"))
    message_play_again = PLAY_AGAIN_FONT.render("Play Again? (Y/N)", 1, pygame.Color("green"))
    SCREEN.blit(message, (320, 240))
    SCREEN.blit(message_play_again, (320 + 12, 240 + 40))

    pygame.display.flip()
    pygame.display.update()

    m_key = get_key()
    while m_key != "exit":
        if m_key == "yes":
            main()
        elif m_key == "no":
            break

        m_key = get_key()
        GAME_CLOCK.tick(FPS)

    sys.exit(0)


def draw_score(score):
    score_numb = SCORE_NUMB_FONT.render(str(score), 1, pygame.Color("red"))
    SCREEN.blit(SCORE_MSG, (SCREEN_WIDTH - SCORE_MSG_SIZE[0] - 60, 10))
    SCREEN.blit(score_numb, (SCREEN_WIDTH - 45, 14))


def draw_game_time(time_of_game):
    game_time = SCORE_FONT.render("Time:", 1, pygame.Color("white"))
    game_time_numb = SCORE_NUMB_FONT.render(str(time_of_game / 1000), 1, pygame.Color("white"))
    SCREEN.blit(game_time, (30, 10))
    SCREEN.blit(game_time_numb, (105, 14))


def respawn_apple(apples, index, sx, sy):
    radius = math.sqrt((SCREEN_WIDTH / 2 * SCREEN_WIDTH / 2 + SCREEN_HEIGHT / 2 * SCREEN_HEIGHT / 2)) / 2
    angle = 999
    x = 0
    y = 0

    while angle > radius:
        angle = random.uniform(0, 800) * math.pi * 2
        x = SCREEN_WIDTH / 2 + radius * math.cos(angle)
        y = SCREEN_HEIGHT / 2 + radius * math.sin(angle)
        if x == sx and y == sy:
            break

    new_apple = Apple(x, y, 1)
    apples[index] = new_apple


def respawn_apples(apples, quantity, sx, sy):
    counter = 0
    del apples[:]
    radius = math.sqrt((SCREEN_WIDTH / 2 * SCREEN_WIDTH / 2 + SCREEN_HEIGHT / 2 * SCREEN_HEIGHT / 2)) / 2
    angle = 999
    x = 0
    y = 0

    while counter < quantity:
        while angle > radius:
            angle = random.uniform(0, 800) * math.pi * 2
            x = SCREEN_WIDTH / 2 + radius * math.cos(angle)
            y = SCREEN_HEIGHT / 2 + radius * math.sin(angle)
            if (x - APPLE_SIZE == sx or x + APPLE_SIZE == sx) and \
                    (y - APPLE_SIZE == sy or y + APPLE_SIZE == sy) or \
                    radius - angle <= 10:
                break

        apples.append(Apple(x, y, 1))
        angle = 999
        counter += 1


def main():
    score = 0

    my_snake = Snake(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)
    my_snake.set_direction(KEY["UP"])
    my_snake.move()
    start_segments = 3

    while start_segments > 0:
        my_snake.grow()
        my_snake.move()
        start_segments -= 1

    max_apples = 1
    eaten_apple = False
    apples = [Apple(random.randint(60, SCREEN_WIDTH), random.randint(60, SCREEN_HEIGHT), 1)]
    respawn_apples(apples, max_apples, my_snake.x, my_snake.y)

    start_time = pygame.time.get_ticks()
    end_of_game = 0

    while end_of_game != 1:
        GAME_CLOCK.tick(FPS)
       
        key_press = get_key()
        if key_press == "exit":
            end_of_game = 1
        
        check_limits(my_snake)
        if my_snake.check_crashing():
            end_game()

        for my_apple in apples:
            if my_apple.state == 1:
                if check_collision(my_snake.get_head(), SNAKE_SIZE, my_apple, APPLE_SIZE):
                    my_snake.grow()
                    my_apple.state = 0
                    score += 10
                    eaten_apple = True
       
        if key_press:
            my_snake.set_direction(key_press)
        my_snake.move()

        if eaten_apple:
            eaten_apple = False
            respawn_apple(apples, 0, my_snake.get_head().x, my_snake.get_head().y)
        
        SCREEN.fill(BACKGROUND_COLOR)
        for my_apple in apples:
            if my_apple.state == 1:
                my_apple.draw(SCREEN)
        
        my_snake.draw(SCREEN)
        draw_score(score)
        game_time = pygame.time.get_ticks() - start_time
        draw_game_time(game_time)

        pygame.display.flip()
        pygame.display.update()


if __name__ == "__main__":
    main()
