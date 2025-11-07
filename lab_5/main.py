import sys
import os
from random import randint
import pygame as pg
import sqlite3
from loguru import logger

# --- Настройка Логирования ---
logger.add("debug.log", format="{time} {level} {message}", level="DEBUG", rotation="10 MB", compression="zip")
logger.add(sys.stdout,
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level>"
                  " | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
           level="INFO")

# --- Константы ---
SCREEN_WIDTH, SCREEN_HEIGHT = 420, 600
BLOCK_SIZE = 60
FPS = 100
BALL_RADIUS = 10
FONT_SIZE = 25
DEFAULT_VOLUME = 0.4
ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))


# --- Классы Игры ---
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


class Button(pg.sprite.Sprite):
    def __init__(self, image_path, hover_image_path, y, level=None, is_back_button=False, *group):
        super().__init__(*group)
        try:
            self.default_image = pg.image.load(os.path.join(ASSETS_DIR, image_path))
            self.hover_image = pg.image.load(os.path.join(ASSETS_DIR, hover_image_path))
        except pg.error as e:
            logger.error(
                f"Не удалось загрузить изображение для кнопки: {image_path} или {hover_image_path}. Ошибка: {e}")
            sys.exit()

        self.image = self.default_image
        self.rect = self.image.get_rect()
        self.rect.x = 101
        self.rect.y = y
        self.level = level
        self.is_back_button = is_back_button

    def update(self, event, game):
        mouse_pos = pg.mouse.get_pos()

        if self.rect.collidepoint(mouse_pos):
            self.image = self.hover_image
        else:
            self.image = self.default_image

        if event and event.type == pg.MOUSEBUTTONUP and self.rect.collidepoint(mouse_pos):
            if not game.game_is_started:
                logger.info(f"Игра началась. Выбран уровень: {self.level}")
                pg.mixer.music.set_volume(game.volume)
                pg.mixer.music.play(99999)
                game.game_is_started = True
                game.levels[self.level] = 1
                game.score = 0
            elif game.game_is_started and self.is_back_button:
                pg.mixer.music.stop()
                game.game_is_started = False
                game.levels = {key: 0 for key in game.levels}
                game.balls = []
                game.should_reset_level = True
                game.score = 0
            elif game.game_is_started and not self.is_back_button:
                game.should_reset_level = True
                pg.mixer.music.set_volume(game.volume)


class Game:
    def __init__(self):
        logger.info("Инициализация игры...")
        pg.init()
        pg.font.init()

        self.screen = pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pg.display.set_caption("Balls and blocks")
        self.font_style = pg.font.SysFont(None, FONT_SIZE)
        self.clock = pg.time.Clock()

        self.balls = []
        self.volume = DEFAULT_VOLUME
        self.music = True
        self.game_is_started = False
        self.levels = {'level_1': 0, 'level_2': 0, 'level_infinity': 0}
        self.game_win = False
        self.block_list = []
        self.color_list = [()] * 70
        self.should_reset_level = False
        self.score = 0
        self.shots_fired = 0
        self.block_was_hit = False

        self._load_sounds()
        self._load_images()
        self._setup_db()
        self._create_objects()
        logger.info("Игра успешно инициализирована.")

    def _load_sounds(self):
        pg.mixer.music.load(os.path.join(ASSETS_DIR, 'LHS-RLD10.mp3'))
        pg.mixer.music.set_volume(self.volume)
        self.sound_shoot = pg.mixer.Sound(os.path.join(ASSETS_DIR, 'shoot.mp3'))
        self.sound_game_over = pg.mixer.Sound(os.path.join(ASSETS_DIR, 'Game over.mp3'))
        self.sound_win = pg.mixer.Sound(os.path.join(ASSETS_DIR, 'winsound.mp3'))
        self.sound_game_over.set_volume(0.2)

    def _load_images(self):
        self.arrow_img = pg.image.load(os.path.join(ASSETS_DIR, "arrow1.png"))
        self.game_background_img = pg.image.load(os.path.join(ASSETS_DIR, '112.jpg')).convert()
        self.menu_background_img = pg.image.load(os.path.join(ASSETS_DIR, 'menu1.png')).convert()
        self.win_screen_img = pg.image.load(os.path.join(ASSETS_DIR, 'win.png'))
        self.game_over_screen_img = pg.image.load(os.path.join(ASSETS_DIR, "game_over1.jpg"))

    def _setup_db(self):
        self.con = sqlite3.connect(os.path.join(ASSETS_DIR, 'records.sqlite3'))
        self.cur = self.con.cursor()
        self.high_scores = self.cur.execute("SELECT счёт from the_best_score").fetchall()
        self.win_counts = self.cur.execute("SELECT num_of_wins from the_best_score").fetchall()

    def _update_win_count_in_db(self):
        active_level_index = list(self.levels.values()).index(1)
        self.cur.execute("UPDATE the_best_score SET num_of_wins = num_of_wins + 1 WHERE номер = ?",
                         (active_level_index + 1,))
        self.con.commit()

    def _update_high_score_in_db(self):
        active_level_index = list(self.levels.values()).index(1)
        if self.score > self.high_scores[active_level_index][0]:
            self.cur.execute("UPDATE the_best_score SET счёт = ? WHERE номер = ?",
                             (self.score, active_level_index + 1))
            self.con.commit()

    def _create_objects(self):
        self.move_event = pg.USEREVENT
        pg.time.set_timer(self.move_event, 2)

        self.arrow_rect = self.arrow_img.get_rect()
        pg.mouse.set_visible(False)

        for i in range(0, SCREEN_HEIGHT, BLOCK_SIZE):
            for j in range(0, SCREEN_WIDTH, BLOCK_SIZE):
                rect = pg.Rect(j, i, BLOCK_SIZE - 1, BLOCK_SIZE - 1)
                self.block_list.append(rect)

        self.x2 = SCREEN_WIDTH // 2
        self.y2 = SCREEN_HEIGHT - 10
        ball_rect_size = int(BALL_RADIUS * 2)
        self.ball = pg.Rect(self.x2, self.y2, ball_rect_size, ball_rect_size)

        self.win_sprite = pg.sprite.Sprite()
        self.win_sprite.image = self.win_screen_img
        self.win_sprite.rect = self.win_sprite.image.get_rect()

        self.gameover_sprite = pg.sprite.Sprite()
        self.gameover_sprite.image = self.game_over_screen_img
        self.gameover_sprite.rect = self.gameover_sprite.image.get_rect()

        self.all_sprites = pg.sprite.Group()
        self.backing = pg.sprite.Group()
        Button('button1.png', 'button1.1.png', 350, 'level_1', False, self.all_sprites)
        Button('button2.png', 'button2.1.png', 426, 'level_2', False, self.all_sprites)
        Button('infinity.png', 'infinity.1.png', 502, 'level_infinity', False, self.all_sprites)
        Button('back.png', 'back1.png', 290, None, False, self.backing)
        Button('menupic.png', 'menupic1.png', 200, None, True, self.backing)

    def message(self, msg, x, y, color, font_style_override=None):
        if font_style_override is None:
            font_style_override = self.font_style
        mesg2 = font_style_override.render(msg, True, color)
        self.screen.blit(mesg2, (x, y))

    def painter(self, colors, blocks, current_shots=0, max_shots=100000000):
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

    def check_win_condition(self, colors, current_shots, max_shots=100000000):
        if not any(colors) and current_shots >= max_shots:
            if not self.game_win:
                logger.info("Условие победы выполнено!")
            self.game_win = True

    def run(self):
        is_running = True
        while is_running:
            if self.game_is_started:
                pg.mouse.set_visible(False)
                if not any(self.color_list[63:]):
                    for event in pg.event.get():
                        if event.type == pg.QUIT:
                            is_running = False

                        elif event.type == pg.MOUSEMOTION:
                            self.arrow_rect.topleft = event.pos

                        elif self.balls and event.type == self.move_event:
                            for b in self.balls:
                                b.move()
                                self.x2 = b.x + 15
                                self.y2 = b.y
                                self.ball.x = self.x2
                                self.ball.y = self.y2
                                if b.y >= SCREEN_HEIGHT - 10:
                                    self.balls = []

                        elif event.type == pg.KEYUP:
                            if event.key == pg.K_1:
                                self.music = not self.music
                                if self.music:
                                    pg.mixer.music.unpause()
                                else:
                                    pg.mixer.music.pause()
                            elif event.key == pg.K_2 and self.volume < 1:
                                self.volume = min(1.0, self.volume + 0.1)
                                pg.mixer.music.set_volume(self.volume)
                            elif event.key == pg.K_3 and self.volume > 0:
                                self.volume = max(0.0, self.volume - 0.1)
                                pg.mixer.music.set_volume(self.volume)
                            elif (event.key == pg.K_2 and self.volume >= 1) or \
                                    (event.key == pg.K_3 and self.volume <= 0):
                                logger.warning(
                                    "Попытка изменить громкость за пределами допустимого диапазона (0.0-1.0)")


                        elif event.type == pg.KEYDOWN and not self.balls:
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
                                    menu_event = pg.event.wait()
                                    if menu_event.type == pg.QUIT:
                                        is_running = False
                                        break
                                    self.backing.update(menu_event, self)
                                    pg.mouse.set_visible(True)
                                    self.screen.fill((0, 0, 0))
                                    self.backing.draw(self.screen)
                                    pg.display.flip()
                                    self.clock.tick(FPS)
                                    if self.should_reset_level:
                                        self.should_reset_level = False
                                        break
                                if not is_running: break

                            if direction_key is not None and not any(self.color_list[63:]):
                                pos = pg.mouse.get_pos()
                                ball_params = [pos[0], SCREEN_HEIGHT - 10, 1, direction_key]
                                logger.debug(f"Создан новый шар с параметрами: {ball_params}")
                                self.balls.append(Ball(self.ball, *ball_params))
                                self.sound_shoot.play()
                                self.shots_fired += 1

                                if self.levels['level_infinity']:
                                    self.color_list = self.painter(self.color_list, self.block_list).copy()
                                elif self.levels['level_1']:
                                    self.color_list = self.painter(self.color_list, self.block_list, self.shots_fired,
                                                                   10).copy()
                                elif self.levels['level_2']:
                                    self.color_list = self.painter(self.color_list, self.block_list, self.shots_fired,
                                                                   20).copy()

                    if self.levels['level_infinity']:
                        self.check_win_condition(self.color_list, self.shots_fired)
                    elif self.levels['level_1']:
                        self.check_win_condition(self.color_list, self.shots_fired, 10)
                    elif self.levels['level_2']:
                        self.check_win_condition(self.color_list, self.shots_fired, 20)

                hit_index = self.ball.collidelist(self.block_list)
                if hit_index != -1 and self.color_list[hit_index]:
                    logger.debug(f"Шар попал в блок с индексом {hit_index}")
                    hit_rect = self.block_list[hit_index]
                    for b in self.balls:
                        b.move(hit_rect)
                    self.color_list[hit_index] = ()
                    self.block_was_hit = True

                self.screen.fill((0, 0, 0))
                self.screen.blit(self.game_background_img, (0, 0))

                if not any(self.color_list[63:]) and not self.game_win:
                    for i, rect in enumerate(self.block_list):
                        if self.color_list[i]:
                            pg.draw.rect(self.screen, self.color_list[i], rect)

                    self.screen.blit(self.arrow_img, self.arrow_rect)
                    pg.draw.circle(self.screen, (255, 250, 250), (self.x2, self.y2), BALL_RADIUS)

                    if hit_index != -1:
                        if self.block_was_hit:
                            self.score += 1
                            self.block_was_hit = False
                        self.message(f'Score: {self.score}', 342, 10, 'snow')
                    pg.display.flip()

                else:
                    if not self.game_win:
                        logger.info(f"Игра окончена. Поражение. Финальный счет: {self.score}")
                        self.sound_game_over.play()
                    else:
                        logger.info(f"Игра окончена. Победа! Финальный счет: {self.score}")
                        self.sound_win.play()
                        self._update_win_count_in_db()
                    pg.mixer.music.stop()

                    self._update_high_score_in_db()

                    end_screen_running = True
                    while end_screen_running:
                        if not self.game_win:
                            self.screen.blit(self.gameover_sprite.image, (0, -100))
                        else:
                            self.screen.blit(self.win_sprite.image, (0, 0))
                        pg.mouse.set_visible(True)
                        self.balls = []
                        for event in pg.event.get():
                            if event.type == pg.QUIT:
                                is_running = False
                                end_screen_running = False
                            elif event.type == pg.KEYDOWN and event.key == pg.K_SPACE:
                                self.game_is_started = False
                                self.shots_fired = 0
                                self.color_list = [()] * 70
                                self.high_scores = self.cur.execute("SELECT счёт from the_best_score").fetchall()
                                self.levels = {key: 0 for key in self.levels}
                                end_screen_running = False

                        pg.display.flip()
                        self.clock.tick(FPS)
            else:
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        is_running = False
                    self.all_sprites.update(event, self)

                pg.mouse.set_visible(True)
                self.screen.fill((0, 0, 0))
                self.screen.blit(self.menu_background_img, (0, 0))
                self.message('Best scores', 40, 235, 'snow', pg.font.SysFont(None, 40))
                self.message(f'1st LEVEL: {self.high_scores[0][0]}', 40, 270, 'gray', pg.font.SysFont(None, 30))
                self.message(f'2nd LEVEL: {self.high_scores[1][0]}', 40, 295, 'gray', pg.font.SysFont(None, 30))
                self.message(f'INFINITY: {self.high_scores[2][0]}', 40, 320, 'gray', pg.font.SysFont(None, 30))
                self.message('Number of wins', 220, 237, 'snow', pg.font.SysFont(None, 33))
                self.message(f'1st LEVEL: {self.win_counts[0][0]}', 220, 270, 'gray', pg.font.SysFont(None, 30))
                self.message(f'2nd LEVEL: {self.win_counts[1][0]}', 220, 295, 'gray', pg.font.SysFont(None, 30))

                if self.score != 0:
                    self.message(f'Your score: {self.score}', 100, 190, 'gold', pg.font.SysFont(None, 50))

                self.all_sprites.draw(self.screen)

            pg.display.flip()
            self.clock.tick(FPS)

        logger.info("Приложение завершает работу.")
        self.con.commit()
        self.con.close()
        pg.quit()
        sys.exit()


if __name__ == "__main__":
    game = Game()
    game.run()
