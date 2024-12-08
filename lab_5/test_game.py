import pytest
import pygame

from pygame.sprite import Group

import game_functions as gf

from game_functions import dump_high_score, load_high_score
from settings import Settings
from button import Button
from game_stats import GameStats
from ship import Ship
from scoreboard import Scoreboard
from paths import TEST_HIGH_SCORE


@pytest.fixture(autouse=True)
def init_game():
    pygame.init()
    yield
    pygame.quit()


@pytest.fixture()
def init_settings():
    settings = Settings()
    screen = pygame.display.set_mode((settings.screen_width,
								   		 settings.screen_height))
    return settings, screen


def test_settings(init_settings):
    settings, _ = init_settings
    assert settings.screen_width == 1280
    assert settings.screen_height == 720
    assert settings.ship_limit == 3
    assert settings.score_scale == 1.5


def test_button(init_settings):
    _, screen = init_settings
    play_button = Button(screen, "Play")
    assert play_button.screen


def test_stats(init_settings):
    settings, _ = init_settings
    stats = GameStats(settings, TEST_HIGH_SCORE)
    assert stats.game_active == False
    assert stats.level == 1
    assert stats.score == 0
    assert stats.high_score != 0 


@pytest.mark.parametrize('moving_left, moving_right, new_center',
                            [(False, True, 641.5),
                             (True, False, 638.5),
                             (False, False, 640.0)])
def test_ship_update(init_settings, moving_left, moving_right, new_center):
    settings, screen = init_settings
    settings.initialize_dyn_settings()
    ship = Ship(screen, settings)
    ship.moving_left = moving_left
    ship.moving_right = moving_right
    ship.update()
    assert ship.center == new_center


@pytest.mark.parametrize('ships_left, game_active',
                            [(3, True),
                             (1, True),
                             (0, False)])
def test_ship_hit(init_settings, ships_left, game_active):
    settings, screen = init_settings
    settings.initialize_dyn_settings()
    aliens = Group()
    bullets = Group()
    stats = GameStats(settings, TEST_HIGH_SCORE)
    stats.game_active = True
    scoreboard = Scoreboard(screen, settings, stats)
    ship = Ship(screen, settings)
    stats.ships_left = ships_left
    gf.ship_hit(aliens, settings, ship, stats, screen, bullets, scoreboard)
    assert stats.game_active == game_active


@pytest.mark.xfail()
def test_collide(init_settings):
    settings, screen = init_settings
    ship = Ship(screen, settings)
    aliens = Group()
    gf.create_fleet(screen, settings, aliens, ship)
    assert pygame.sprite.spritecollideany(ship, aliens)


def test_load_dump_high_score(init_settings):
    settings, _ = init_settings
    stats = GameStats(settings, TEST_HIGH_SCORE)
    assert stats.high_score == 10000
    stats.high_score = 25000
    dump_high_score(stats, TEST_HIGH_SCORE)
    loaded_high_score = load_high_score(TEST_HIGH_SCORE)
    assert loaded_high_score == 25000
    stats.high_score = 10000
    dump_high_score(stats, TEST_HIGH_SCORE)


def test_center_ship(init_settings):
    settings, screen = init_settings
    ship = Ship(screen, settings)
    assert ship.rect.centerx == screen.get_rect().centerx
    assert ship.rect.bottom == screen.get_rect().bottom 


def test_create_stars(init_settings):
    settings, screen = init_settings
    stars = Group()
    gf.create_stars(screen, settings, stars)
    assert len(stars) > 0


def test_create_fleet(init_settings):
    settings, screen = init_settings
    ship = Ship(screen, settings)
    aliens = Group()
    gf.create_fleet(screen, settings, aliens, ship)
    assert len(aliens) > 0 
   