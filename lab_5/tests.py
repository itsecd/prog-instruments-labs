import pytest

from player import Player
from enemy import Enemy


def test_player_position():
    player = Player()
    assert player.rect.x == 400
    assert player.rect.y == 300


def test_player_move():
    player = Player()
    player.move(10, 0)
    assert player.rect.x == 410


def test_player_dont_move_through_border():
    player = Player()
    player.move(1000, 0)
    assert player.rect.x == 750


def test_player_dont_move_through_border_twice():
    player = Player()
    player.move(-5000, 0)
    assert player.rect.x == 0

    player.move(10000, 0)
    assert player.rect.x == 750


def test_enemy_position():
    enemy = Enemy()
    assert 0 <= enemy.rect.x <= 800
    assert 0 <= enemy.rect.y <= 600


def test_kill_enemy():
    player = Player()
    enemy = Enemy()
    enemy.rect.x = player.rect.x
    enemy.rect.y = player.rect.y
    assert player.rect.colliderect(enemy.rect)


def test_kill_enemy_and_remove_from_list():
    player = Player()
    enemy = Enemy()
    enemy.rect.x = player.rect.x
    enemy.rect.y = player.rect.y
    enemies = [enemy]

    if player.rect.colliderect(enemy.rect):
        enemies.remove(enemy)
    assert len(enemies) == 0


def test_kill_all_enemies_and_remove_from_list():
    player = Player()
    enemies = [Enemy() for _ in range(5)]
    player.rect.x = 400
    player.rect.y = 300

    for enemy in enemies:
        enemy.rect.x = player.rect.x
        enemy.rect.y = player.rect.y

    for enemy in enemies[:]:
        if player.rect.colliderect(enemy.rect):
            enemies.remove(enemy)

    assert len(enemies) == 0

