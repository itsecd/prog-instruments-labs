import pytest
import pygame

from player import Player


def player_position_test():
    player = Player()
    assert player.rect.x == 400
    assert player.rect.y == 300
