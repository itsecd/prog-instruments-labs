import random

import pygame

from config import WIDTH, HEIGHT, ENEMY_SIZE


class Enemy:
    def __init__(self):
        self.rect = pygame.Rect(random.randint(0, WIDTH - ENEMY_SIZE),
                                random.randint(0, HEIGHT - ENEMY_SIZE), ENEMY_SIZE, ENEMY_SIZE)
