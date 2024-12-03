import pygame

from laser import Laser


class Spaceship(pygame.sprite.Sprite):

	def __init__(self, screen_width: int, screen_height: int, offset: int) -> None:
		"""Create an object of the Spaceship class"""
		super().__init__()
		self.offset = offset
		self.screen_width = screen_width
		self.screen_height = screen_height
		self.image = pygame.image.load("lab_1/Graphics/spaceship.png")
		self.rect = self.image.get_rect(midbottom=((self.screen_width + self.offset)/2, self.screen_height))
		self.speed = 6
		self.lasers_group = pygame.sprite.Group()
		self.laser_ready = True
		self.laser_time = 0
		self.laser_delay = 300
		self.laser_sound = pygame.mixer.Sound("lab_1/Sounds/laser.ogg")

	def get_user_input(self) -> None:
		"""Get the user's keystroke"""
		keys = pygame.key.get_pressed()

		if keys[pygame.K_RIGHT]:
			self.rect.x += self.speed

		if keys[pygame.K_LEFT]:
			self.rect.x -= self.speed

		if keys[pygame.K_SPACE] and self.laser_ready:
			self.laser_ready = False
			laser = Laser(self.rect.center, 5, self.screen_height)
			self.lasers_group.add(laser)
			self.laser_time = pygame.time.get_ticks()
			self.laser_sound.play()

	def update(self) -> None:
		"""Update Spaceship status"""
		self.get_user_input()
		self.constrain_movement()
		self.lasers_group.update()
		self.recharge_laser()

	def constrain_movement(self) -> None:
		"""A method to ensure that the spacecraft does not go beyond the window"""
		if self.rect.right > self.screen_width:
			self.rect.right = self.screen_width
		if self.rect.left < self.offset:
			self.rect.left = self.offset

	def recharge_laser(self) -> None:
		"""Recharge the laser"""
		if not self.laser_ready:
			current_time = pygame.time.get_ticks()
			if current_time - self.laser_time >= self.laser_delay:
				self.laser_ready = True

	def reset(self) -> None:
		"""Update the location of the spaceship on the playing field"""
		self.rect = self.image.get_rect(midbottom=((self.screen_width + self.offset)/2, self.screen_height))
		self.lasers_group.empty()