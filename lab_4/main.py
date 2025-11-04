import sys
import os
from random import randint
import pygame as pg
import sqlite3

# --- Константы ---
SCREEN_WIDTH, SCREEN_HEIGHT = 420, 600
BLOCK_SIZE = 60
FPS = 100
BALL_RADIUS = 10
FONT_SIZE = 25
DEFAULT_VOLUME = 0.4
ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Инициализация ---
pg.init()
pg.font.init()

# --- Настройка Экрана и Шрифта ---
screen = pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pg.display.set_caption("Balls and blocks")
font_style = pg.font.SysFont(None, FONT_SIZE)

# --- Глобальные переменные состояния (временно) ---
game_over = False
balls = []
volume = DEFAULT_VOLUME
music = True
game_is_started = False
levels = {'level_1': 0, 'level_2': 0, 'level_infinity': 0}
game_win = False
block_list = []
color_list = [()] * 70

# --- Настройка звука ---
pg.mixer.music.load(os.path.join(ASSETS_DIR, 'LHS-RLD10.mp3'))
pg.mixer.music.set_volume(volume)
sound_shoot = pg.mixer.Sound(os.path.join(ASSETS_DIR, 'shoot.mp3'))
sound_game_over = pg.mixer.Sound(os.path.join(ASSETS_DIR, 'Game over.mp3'))
sound_win = pg.mixer.Sound(os.path.join(ASSETS_DIR, 'winsound.mp3'))
sound_game_over.set_volume(0.2)

# --- Настройка времени и событий ---
clock = pg.time.Clock()
move = pg.USEREVENT
pg.time.set_timer(move, 2)

# --- Загрузка изображений ---
arrow_img = pg.image.load(os.path.join(ASSETS_DIR, "arrow1.png"))
game_background_img = pg.image.load(os.path.join(ASSETS_DIR, '112.jpg')).convert()
menu_background_img = pg.image.load(os.path.join(ASSETS_DIR, 'menu1.png')).convert()
win_screen_img = pg.image.load(os.path.join(ASSETS_DIR, 'win.png'))
game_over_screen_img = pg.image.load(os.path.join(ASSETS_DIR, "game_over1.jpg"))

# --- Создание игровых объектов ---
arrow_rect = arrow_img.get_rect()
pg.mouse.set_visible(False)

# Создание списка блоков
for i in range(0, SCREEN_HEIGHT, BLOCK_SIZE):
    for j in range(0, SCREEN_WIDTH, BLOCK_SIZE):
        rect = pg.Rect(j, i, BLOCK_SIZE - 1, BLOCK_SIZE - 1)
        block_list.append(rect)

# Инициализация шара
x2 = SCREEN_WIDTH // 2
y2 = SCREEN_HEIGHT - 10
ball_rect_size = int(BALL_RADIUS * 2)
ball = pg.Rect(x2, y2, ball_rect_size, ball_rect_size)

# --- Настройка Базы Данных ---
con = sqlite3.connect(os.path.join(ASSETS_DIR, 'records.sqlite3'))
cur = con.cursor()

# --- Создание спрайтов для экранов ---
win_sprite = pg.sprite.Sprite()
win_sprite.image = win_screen_img
win_sprite.rect = win_sprite.image.get_rect()

gameover_sprite = pg.sprite.Sprite()
gameover_sprite.image = game_over_screen_img
gameover_sprite.rect = gameover_sprite.image.get_rect()


class Button(pg.sprite.Sprite):
    def __init__(self, image_path, hover_image_path, y, level=None, is_back_button=False, *group):
        super().__init__(*group)
        self.default_image = pg.image.load(os.path.join(ASSETS_DIR, image_path))
        self.hover_image = pg.image.load(os.path.join(ASSETS_DIR, hover_image_path))

        self.image = self.default_image
        self.rect = self.image.get_rect()
        self.rect.x = 101
        self.rect.y = y
        self.level = level
        self.is_back_button = is_back_button

    def update(self, *args):
        mouse_pos = pg.mouse.get_pos()
        global game_is_started, levels, score, should_reset_level, balls

        if args and self.rect.collidepoint(mouse_pos):
            self.image = self.hover_image
        else:
            self.image = self.default_image

        if args and args[0].type == pg.MOUSEBUTTONUP and self.rect.collidepoint(mouse_pos):
            if not game_is_started:
                pg.mixer.music.set_volume(volume)
                pg.mixer.music.play(99999)
                game_is_started = True
                levels[self.level] = 1
                score = 0
            elif game_is_started and self.is_back_button:
                pg.mixer.music.stop()
                game_is_started = False
                levels = {key: 0 for key in levels}
                balls = []
                should_reset_level = True
                score = 0
            elif game_is_started and not self.is_back_button:
                should_reset_level = True
                pg.mixer.music.set_volume(volume)


all_sprites = pg.sprite.Group()
backing = pg.sprite.Group()
Button('button1.png', 'button1.1.png', 350, 'level_1', False, all_sprites)
Button('button2.png', 'button2.1.png', 426, 'level_2', False, all_sprites)
Button('infinity.png', 'infinity.1.png', 502, 'level_infinity', False, all_sprites)
Button('back.png', 'back1.png', 290, None, False, backing)
Button('menupic.png', 'menupic1.png', 200, None, True, backing)


class Ball:
    def __init__(self, obj, x, y, speed, direction_mod):
        self.obj = obj
        self.x = x
        self.y = y
        self.dx = -speed * direction_mod
        self.dy = -speed

    def move(self, block=None):
        if block:
            if self.dx > 0:
                delta_x = self.obj.right - block.left
            else:
                delta_x = block.right - self.obj.left
            if self.dy > 0:
                delta_y = self.obj.bottom - block.top
            else:
                delta_y = block.bottom - self.obj.top

            if abs(delta_x - delta_y) < 10:
                self.dx = -self.dx
                self.dy = -self.dy
            elif delta_x > delta_y:
                self.dy = -self.dy
            elif delta_y > delta_x:
                self.dx = -self.dx
        else:
            if self.obj.centerx < BALL_RADIUS or self.obj.centerx > SCREEN_WIDTH - BALL_RADIUS:
                self.dx = -self.dx
            if self.obj.centery < BALL_RADIUS:
                self.dy = -self.dy

        self.x += self.dx
        self.y += self.dy


high_scores = cur.execute("SELECT счёт from the_best_score").fetchall()
win_counts = cur.execute("SELECT num_of_wins from the_best_score").fetchall()


def message(msg, x, y, color, font_style_override=font_style):
    mesg2 = font_style_override.render(msg, True, color)
    screen.blit(mesg2, (x, y))


def painter(colors, blocks, current_shots=0, max_shots=100000000):
    global game_win
    if not any(colors[63:]) and (any(colors[:7]) or current_shots > max_shots):
        lister = colors.copy()
        colors.clear()
        colors = lister[63:]
        colors.extend(lister[:63])
        lister.clear()
    if current_shots <= max_shots:
        for _ in range(5):
            for i, rect1 in enumerate(blocks[:7]):
                if rect1.collidepoint(randint(0, SCREEN_WIDTH), 58):
                    if not colors[i]:
                        colors[i] = (randint(1, 255), randint(1, 255), randint(1, 255))
    return colors


def check_win_condition(colors, current_shots, max_shots=100000000):
    global game_win
    if not any(colors) and current_shots >= max_shots:
        game_win = True


should_reset_level = False
score = 0
shots_fired = 0
block_was_hit = False

while not game_over:
    if game_is_started:
        pg.mouse.set_visible(False)
        if not any(color_list[63:]):
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    con.commit()
                    con.close()
                    sys.exit()

                elif event.type == pg.MOUSEMOTION:
                    arrow_rect.topleft = event.pos

                elif balls and event.type == move:
                    for b in balls:
                        b.move()
                        x2 = b.x + 15
                        y2 = b.y
                        ball.x = x2
                        ball.y = y2
                        if b.y >= SCREEN_HEIGHT - 10:
                            balls = []

                elif event.type == pg.KEYUP:
                    if event.key == pg.K_1:
                        music = not music
                        if music:
                            pg.mixer.music.unpause()
                        else:
                            pg.mixer.music.pause()
                    elif event.key == pg.K_2 and volume < 1:
                        volume = min(1.0, volume + 0.1)
                        pg.mixer.music.set_volume(volume)
                    elif event.key == pg.K_3 and volume > 0:
                        volume = max(0.0, volume - 0.1)
                        pg.mixer.music.set_volume(volume)

                elif event.type == pg.KEYDOWN and not balls:
                    direction_key = None
                    if event.key == pg.K_LEFT:
                        direction_key = 1
                    elif event.key == pg.K_RIGHT:
                        direction_key = -1
                    elif event.key == pg.K_UP:
                        direction_key = 0
                    elif event.key == pg.K_SPACE:
                        while True:
                            pg.mixer.music.set_volume(0.1)
                            for event_2 in pg.event.get():
                                if event_2.type == pg.QUIT:
                                    sys.exit()
                                backing.update(event_2)
                                pg.mouse.set_visible(True)
                                screen.fill((0, 0, 0))
                                backing.draw(screen)

                            pg.display.flip()
                            clock.tick(FPS)
                            if should_reset_level:
                                should_reset_level = False
                                break

                    if direction_key is not None and not any(color_list[63:]):
                        pos = pg.mouse.get_pos()
                        ball_params = [pos[0], SCREEN_HEIGHT - 10, 1, direction_key]
                        balls.append(Ball(ball, *ball_params))
                        sound_shoot.play()
                        shots_fired += 1

                        if levels['level_infinity']:
                            color_list = painter(color_list, block_list).copy()
                        elif levels['level_1']:
                            color_list = painter(color_list, block_list, shots_fired, 10).copy()
                        elif levels['level_2']:
                            color_list = painter(color_list, block_list, shots_fired, 20).copy()

            if levels['level_infinity']:
                check_win_condition(color_list, shots_fired)
            elif levels['level_1']:
                check_win_condition(color_list, shots_fired, 10)
            elif levels['level_2']:
                check_win_condition(color_list, shots_fired, 20)

        hit_index = ball.collidelist(block_list)
        if hit_index != -1 and color_list[hit_index]:
            hit_rect = block_list[hit_index]
            for b in balls:
                b.move(hit_rect)
            color_list[hit_index] = ()
            block_was_hit = True

        screen.fill((0, 0, 0))
        screen.blit(game_background_img, (0, 0))

        if not any(color_list[63:]) and not game_win:
            for i, rect in enumerate(block_list):
                if color_list[i]:
                    pg.draw.rect(screen, color_list[i], rect)

            # --- УПРОЩЕННАЯ ЛОГИКА ---
            # Было: replaced_text = ((str(arrow_rect).replace('<rect(', ''))...
            # Стало:
            screen.blit(arrow_img, arrow_rect)  # <-- Pygame может рисовать прямо по Rect
            # ---------------------------

            pg.draw.circle(screen, (255, 250, 250), (x2, y2), BALL_RADIUS)

            if hit_index != -1:
                if block_was_hit:
                    score += 1
                    block_was_hit = False
                message(f'Score: {score}', 342, 10, 'snow')
            pg.display.flip()

        else:
            if not game_win:
                sound_game_over.play()
            else:
                sound_win.play()
                active_level_index = list(levels.values()).index(1)
                cur.execute("UPDATE the_best_score SET num_of_wins = num_of_wins + 1 WHERE номер = ?",
                            (active_level_index + 1,))
                con.commit()
            pg.mixer.music.stop()

            active_level_index = list(levels.values()).index(1)
            if score > high_scores[active_level_index][0]:
                cur.execute("UPDATE the_best_score SET счёт = ? WHERE номер = ?", (score, active_level_index + 1))
                con.commit()

            while True:
                if not game_win:
                    screen.blit(gameover_sprite.image, (0, -100))
                else:
                    screen.blit(win_sprite.image, (0, 0))
                pg.mouse.set_visible(True)
                balls = []
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        con.commit()
                        con.close()
                        sys.exit()
                    elif event.type == pg.KEYDOWN and event.key == pg.K_SPACE:
                        game_is_started = False
                        shots_fired = 0
                        color_list = [()] * 70
                        high_scores = cur.execute("SELECT счёт from the_best_score").fetchall()
                        levels = {key: 0 for key in levels}

                if not game_is_started:
                    break
                pg.display.flip()
                clock.tick(FPS)
    else:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                sys.exit()
            all_sprites.update(event)

        pg.mouse.set_visible(True)
        screen.fill((0, 0, 0))
        screen.blit(menu_background_img, (0, 0))
        message('Best scores', 40, 235, 'snow', pg.font.SysFont(None, 40))
        message(f'1st LEVEL: {high_scores[0][0]}', 40, 270, 'gray', pg.font.SysFont(None, 30))
        message(f'2nd LEVEL: {high_scores[1][0]}', 40, 295, 'gray', pg.font.SysFont(None, 30))
        message(f'INFINITY: {high_scores[2][0]}', 40, 320, 'gray', pg.font.SysFont(None, 30))
        message('Number of wins', 220, 237, 'snow', pg.font.SysFont(None, 33))
        message(f'1st LEVEL: {win_counts[0][0]}', 220, 270, 'gray', pg.font.SysFont(None, 30))
        message(f'2nd LEVEL: {win_counts[1][0]}', 220, 295, 'gray', pg.font.SysFont(None, 30))

        if score != 0:
            message(f'Your score: {score}', 100, 190, 'gold', pg.font.SysFont(None, 50))

        all_sprites.draw(screen)

    pg.display.flip()
    clock.tick(FPS)
    pg.display.update()
    clock.tick(FPS)
