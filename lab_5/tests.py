import pytest

from player import Player


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

