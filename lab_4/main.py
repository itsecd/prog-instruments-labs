import sys
import os # <-- Добавлен импорт
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
# --- Путь к папке с ресурсами ---
# Это позволяет запускать скрипт из любой директории
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
sound1 = pg.mixer.Sound(os.path.join(ASSETS_DIR, 'shoot.mp3'))
sound2 = pg.mixer.Sound(os.path.join(ASSETS_DIR, 'Game over.mp3'))
sound3 = pg.mixer.Sound(os.path.join(ASSETS_DIR, 'winsound.mp3'))
sound2.set_volume(0.2)

# --- Настройка времени и событий ---
clock = pg.time.Clock()
move = pg.USEREVENT
pg.time.set_timer(move, 2)

# --- Загрузка изображений ---
arrow = pg.image.load(os.path.join(ASSETS_DIR, "arrow1.png"))
img = pg.image.load(os.path.join(ASSETS_DIR, '112.jpg')).convert()
img2 = pg.image.load(os.path.join(ASSETS_DIR, 'menu1.png')).convert()
image1 = pg.image.load(os.path.join(ASSETS_DIR, 'win.png'))
image_game_over = pg.image.load(os.path.join(ASSETS_DIR, "game_over1.jpg"))

# --- Создание игровых объектов ---
r = arrow.get_rect()
pg.mouse.set_visible(False)

# Создание списка блоков
for i in range(0, SCREEN_HEIGHT, BLOCK_SIZE):
    for j in range(0, SCREEN_WIDTH, BLOCK_SIZE):
        rect = pg.Rect(j, i, BLOCK_SIZE - 1, BLOCK_SIZE - 1)
        block_list.append(rect)

# Инициализация шара
x2 = SCREEN_WIDTH // 2
y2 = SCREEN_HEIGHT - 10
ball_rect = int(BALL_RADIUS * 2)
ball = pg.Rect(x2, y2, ball_rect, ball_rect)

# --- Настройка Базы Данных ---
con = sqlite3.connect(os.path.join(ASSETS_DIR, 'records.sqlite3'))
cur = con.cursor()

# --- Создание спрайтов для экранов ---
win = pg.sprite.Sprite()
win.image = image1
win.rect = win.image.get_rect()

gameover = pg.sprite.Sprite()
gameover.image = image_game_over
gameover.rect = gameover.image.get_rect()

class Button(pg.sprite.Sprite):
    def __init__(self, imeg, imeg2, y, level=None, back=0, *group, ):
        super().__init__(*group)
        self.image = pg.Surface((66, 218))
        self.imeg = pg.image.load(os.path.join(ASSETS_DIR, imeg))
        self.image = self.imeg
        self.rect = self.image.get_rect()
        self.rect.x = 101
        self.rect.y = y
        self.level = level
        self.img = pg.image.load(os.path.join(ASSETS_DIR, imeg2))
        self.back = back

    def update(self, *args):
        g = pg.mouse.get_pos()
        global game_is_started
        global levels
        global k
        global re
        global balls
        if args and self.rect.collidepoint(g):
            self.image = self.img
        else:
            self.image = self.imeg

        if args and args[0].type == pg.MOUSEBUTTONUP and \
                self.rect.collidepoint(g) and not game_is_started:
            pg.mixer.music.set_volume(volume)
            pg.mixer.music.play(99999)
            game_is_started = True
            levels[self.level] = 1
            k = 0

        elif args and args[0].type == pg.MOUSEBUTTONUP and \
                self.rect.collidepoint(g) and game_is_started and self.back:
            pg.mixer.music.stop()
            game_is_started = False
            levels['level_1'] = 0
            levels['level_2'] = 0
            levels['level_infinity'] = 0
            balls = []
            re = 1
            k = 0

        elif args and args[0].type == pg.MOUSEBUTTONUP and \
                self.rect.collidepoint(g) and game_is_started and not self.back:
            re = 1
            pg.mixer.music.set_volume(volume)

all_sprites = pg.sprite.Group()
backing = pg.sprite.Group()
Button('button1.png', 'button1.1.png', 350, 'level_1', 0, all_sprites)
Button('button2.png', 'button2.1.png', 426, 'level_2', 0, all_sprites)
Button('infinity.png', 'infinity.1.png', 502, 'level_infinity', 0, all_sprites)
Button('back.png', 'back1.png', 290, None, 0, backing)
Button('menupic.png', 'menupic1.png', 200, None, 1, backing)

class Ball:
    def __init__(self, obj, x, y, v, t):
        self.obj = obj
        self.x = x
        self.y = y
        self.dx = -v * t
        self.dy = -v

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

result = cur.execute("""
SELECT счёт from the_best_score""").fetchall()
result1 = cur.execute("""
SELECT num_of_wins from the_best_score""").fetchall()

def message(msg, x, y, color, font_style1=font_style):
    mesg2 = font_style1.render(msg, True, color)
    screen.blit(mesg2, (x, y))

def painter(colors, blocks, n1=0, n_max=100000000):
    global game_win
    if not any(colors[63:]) and (any(colors[:7]) or n1 > n_max):
        lister = colors.copy()
        colors.clear()
        colors = lister[63:]
        colors.extend(lister[:63])
        lister.clear()
    if n1 <= n_max:
        for le in range(5):
            for g, rect1 in enumerate(blocks[:7]):
                if rect1.collidepoint(randint(0, SCREEN_WIDTH), 58):
                    if not colors[g]:
                        colors[g] = (randint(1, 255), randint(1, 255), randint(1, 255))
    return colors

def if_win(colors, n1, n_max=100000000):
    global game_win
    if not any(colors) and n1 >= n_max:
        game_win = True

re = 0
k = 0
n = 0
d = 0

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
                    r.topleft = event.pos

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
                        if music:
                            pg.mixer.music.pause()
                            music = False
                        else:
                            pg.mixer.music.unpause()
                            music = True
                    elif event.key == pg.K_2 and volume < 1:
                        if volume < 1:
                            volume += 0.1
                            pg.mixer.music.set_volume(volume)
                    elif event.key == pg.K_3 and volume > 0:
                        if volume > 0:
                            volume -= 0.1
                            pg.mixer.music.set_volume(volume)

                elif event.type == pg.KEYDOWN and balls == []:
                    KEY = 2
                    if event.key == pg.K_LEFT:
                        KEY = 1
                    elif event.key == pg.K_RIGHT:
                        KEY = -1
                    elif event.key == pg.K_UP:
                        KEY = 0
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
                            if re == 1:
                                re = 0
                                break

                    if KEY != 2 and not any(color_list[63:]):
                        pos = pg.mouse.get_pos()
                        a = [pos[0], SCREEN_HEIGHT - 10, 1]
                        balls.append(Ball(ball, *a, KEY))
                        sound1.play()
                        n += 1

                        if levels['level_infinity']:
                            color_list = painter(color_list, block_list).copy()
                        elif levels['level_1']:
                            color_list = painter(color_list, block_list, n, 10).copy()
                        elif levels['level_2']:
                            color_list = painter(color_list, block_list, n, 20).copy()
            if levels['level_infinity']:
                if_win(color_list, n)
            elif levels['level_1']:
                if_win(color_list, n, 10)
            elif levels['level_2']:
                if_win(color_list, n, 20)

        hit_index = ball.collidelist(block_list)
        if hit_index != -1:
            if color_list[hit_index]:
                hit_rect = block_list[hit_index]
                for b in balls:
                    b.move(hit_rect)
                color_list[hit_index] = ()
                d = 1
        screen.fill((0, 0, 0))

        screen.blit(img, (0, 0))
        if not any(color_list[63:]) and not game_win:

            for i, rect in enumerate(block_list):
                if color_list[i] != ():
                    pg.draw.rect(screen, color_list[i], rect)

            replaced_text = ((str(r).replace('<rect(', '')).replace(')>', '')).split(", ")
            x1, y1 = int(replaced_text[0]), int(replaced_text[1])
            screen.blit(arrow, (x1, y1))

            pg.draw.circle(screen, (255, 250, 250), (x2, y2), BALL_RADIUS)

            if hit_index != -1:
                if d:
                    k += 1
                    d = 0
                message(f'Score: {k}', 342, 10, 'snow')
            clock.tick(FPS)
            pg.display.flip()

        else:
            if not game_win:
                sound2.play()
            else:
                sound3.play()
                cur.execute("""UPDATE the_best_score
                    SET num_of_wins = num_of_wins + 1 WHERE номер = {}""".format(list(levels.values()).index(1) + 1)).fetchall()
                con.commit()
            pg.mixer.music.stop()
            if k > result[list(levels.values()).index(1)][0]:
                cur.execute("""UPDATE the_best_score
                    SET счёт = {} WHERE номер = {}""".format(k, list(levels.values()).index(1) + 1)).fetchall()
                con.commit()
            while True:
                if not game_win:
                    screen.blit(gameover.image, (0, -100))
                else:
                    screen.blit(win.image, (0, 0))
                pg.mouse.set_visible(True)
                balls = []
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        con.commit()
                        con.close()
                        sys.exit()

                    elif event.type == pg.KEYDOWN:
                        if event.key == pg.K_SPACE:
                            game_is_started = False
                            n = 0
                            color_list = [()] * 70
                            result = cur.execute("""
                            SELECT счёт from the_best_score""").fetchall()
                            levels['level_infinity'] = 0
                            levels['level_1'] = 0
                            levels['level_2'] = 0
                pg.display.flip()
                clock.tick(FPS)
                if not game_is_started:
                    break
    else:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                sys.exit()
            all_sprites.update(event)

        pg.mouse.set_visible(True)
        screen.fill((0, 0, 0))
        screen.blit(img2, (0, 0))
        message('Best scores', 40, 235, 'snow', pg.font.SysFont(None, 40))
        message(f'1st LEVEL: {result[0][0]}', 40, 270, 'gray', pg.font.SysFont(None, 30))
        message(f'2nd LEVEL: {result[1][0]}', 40, 295, 'gray', pg.font.SysFont(None, 30))
        message(f'INFINITY: {result[2][0]}', 40, 320, 'gray', pg.font.SysFont(None, 30))
        message('Number of wins', 220, 237, 'snow', pg.font.SysFont(None, 33))
        message(f'1st LEVEL: {result1[0][0]}', 220, 270, 'gray', pg.font.SysFont(None, 30))
        message(f'2nd LEVEL: {result1[1][0]}', 220, 295, 'gray', pg.font.SysFont(None, 30))
        if k != 0:
            message(f'Your score: {k}', 100, 190, 'gold', pg.font.SysFont(None, 50))
        all_sprites.draw(screen)

    pg.display.flip()
    clock.tick(FPS)
    pg.display.update()
    clock.tick(FPS)
