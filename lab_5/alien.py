import pygame

from pygame.sprite import Sprite

from paths import ALIEN_IMAGE


class Alien(Sprite):
	"""A class that implements the alien's behavior"""

	def __init__(self, screen, ai_settings):
		"""Initializes the alien and sets its initial position"""
		super().__init__()
		self.screen = screen
		self.ai_settings = ai_settings
		self.image = pygame.image.load(ALIEN_IMAGE)
		self.rect = self.image.get_rect()
		self.rect.x = self.rect.width
		self.rect.y = self.rect.height
		self.x = float(self.rect.x)
		self.y = float(self.rect.y)

	def blitme(self):
		"""Draws the alien in the current position"""
		self.screen.blit(self.image, self.rect)

	def update(self):
		"""Updates the position of the alien"""
		self.x += (self.ai_settings.alien_speed_factor*self.ai_settings.fleet_direction)
		self.rect.x = self.x

	def check_edges(self):
		"""Returns True if the alien is at the edge of the screen"""
		if self.rect.right >= self.screen.get_rect().right:
			return True
		elif self.rect.left <= 0:
			return True
 