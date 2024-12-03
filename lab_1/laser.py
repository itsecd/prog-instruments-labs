import pygame


class Laser(pygame.sprite.Sprite):

	def __init__(self, position: int, speed: int, screen_height: int) -> None:
		"""Create an object of the Laser class"""
		super().__init__()
		self.image = pygame.Surface((4, 15))
		self.image.fill((243, 216, 63))
		self.rect = self.image.get_rect(center = position)
		self.speed = speed
		self.screen_height = screen_height

	def update(self) -> None:
		"""Update laser shot status"""
		self.rect.y -= self.speed
		if self.rect.y > self.screen_height + 15 or self.rect.y < 0:
			self.kill()