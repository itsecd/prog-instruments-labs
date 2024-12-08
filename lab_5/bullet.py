import pygame

from pygame.sprite import Sprite


class Bullet(Sprite):
	"""A class for controlling bullets fired by a ship"""

	def __init__(self, ai_settings, screen, ship):
		"""Creates a bullet object at the current position of the ship"""
		super().__init__()
		self.screen = screen
		self.rect = pygame.Rect(0, 0, ai_settings.bullet_width,
						  		 ai_settings.bullet_height)
		self.rect.centerx = ship.rect.centerx
		self.rect.top = ship.rect.top
		self.y = float(self.rect.y)
		self.color = ai_settings.bullet_color
		self.speed_factor = ai_settings.bullet_speed_factor

	def update(self):
		"""Moves the bullet across the screen"""
		self.y -= self.speed_factor
		self.rect.y = self.y

	def draw_bullet(self):
		"""Displaying a bullet on the screen"""
		pygame.draw.rect(self.screen, self.color, self.rect)
 