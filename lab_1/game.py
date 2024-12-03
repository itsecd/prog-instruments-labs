import random

import pygame

from alien import Alien, MysteryShip
from laser import Laser
from obstacle import Obstacle, grid
from spaceship import Spaceship


class Game:

	def __init__(self, screen_width: int, screen_height: int, offset: int) -> None:
		"""Create an object of the Game class"""
		self.screen_width = screen_width
		self.screen_height = screen_height
		self.offset = offset
		self.spaceship_group = pygame.sprite.GroupSingle()
		self.spaceship_group.add(Spaceship(self.screen_width, self.screen_height, self.offset))
		self.obstacles = self.create_obstacles()
		self.aliens_group = pygame.sprite.Group()
		self.create_aliens()
		self.aliens_direction = 1
		self.alien_lasers_group = pygame.sprite.Group()
		self.mystery_ship_group = pygame.sprite.GroupSingle()
		self.lives = 3
		self.run = True
		self.score = 0
		self.highscore = 0
		self.explosion_sound = pygame.mixer.Sound("lab_1/Sounds/explosion.ogg")
		self.load_highscore()
		pygame.mixer.music.load("lab_1/Sounds/music.ogg")
		pygame.mixer.music.play(-1)

	def create_obstacles(self) -> None:
		"""Create obstacles on the playing field"""
		obstacle_width = len(grid[0]) * 3
		gap = (self.screen_width + self.offset - (4 * obstacle_width))/5
		obstacles = []
		for i in range(4):
			offset_x = (i + 1) * gap + i * obstacle_width
			obstacle = Obstacle(offset_x, self.screen_height - 100)
			obstacles.append(obstacle)
		return obstacles

	def create_aliens(self) -> None:
		"""Create aliens on the playing field"""
		for row in range(5):
			for column in range(11):
				x = 75 + column * 55
				y = 110 + row * 55

				if row == 0:
					alien_type = 3
				elif row in (1, 2):
					alien_type = 2
				else:
					alien_type = 1

				alien = Alien(alien_type, x + self.offset/2, y)
				self.aliens_group.add(alien)

	def move_aliens(self) -> None:
		"""Moves the alien ship"""
		self.aliens_group.update(self.aliens_direction)

		alien_sprites = self.aliens_group.sprites()
		for alien in alien_sprites:
			if alien.rect.right >= self.screen_width + self.offset/2:
				self.aliens_direction = -1
				self.alien_move_down(2)
			elif alien.rect.left <= self.offset/2:
				self.aliens_direction = 1
				self.alien_move_down(2)

	def alien_move_down(self, distance: None) -> None:
		"""Moves the alien ship down"""
		if self.aliens_group:
			for alien in self.aliens_group.sprites():
				alien.rect.y += distance

	def alien_shoot_laser(self) -> None:
		"""Creates a shot by an alien ship"""
		if self.aliens_group.sprites():
			random_alien = random.choice(self.aliens_group.sprites())
			laser_sprite = Laser(random_alien.rect.center, -6, self.screen_height)
			self.alien_lasers_group.add(laser_sprite)

	def create_mystery_ship(self) -> None:
		"""Create mystery ship :)"""
		self.mystery_ship_group.add(MysteryShip(self.screen_width, self.offset))

	def check_for_collisions(self) -> None:
		"""Checks for laser hits"""
		#Spaceship
		if self.spaceship_group.sprite.lasers_group:
			for laser_sprite in self.spaceship_group.sprite.lasers_group:
				
				aliens_hit = pygame.sprite.spritecollide(laser_sprite, self.aliens_group, True)
				if aliens_hit:
					self.explosion_sound.play()
					for alien in aliens_hit:
						self.score += alien.type * 100
						self.check_for_highscore()
						laser_sprite.kill()

				if pygame.sprite.spritecollide(laser_sprite, self.mystery_ship_group, True):
					self.score += 500
					self.explosion_sound.play()
					self.check_for_highscore()
					laser_sprite.kill()

				for obstacle in self.obstacles:
					if pygame.sprite.spritecollide(laser_sprite, obstacle.blocks_group, True):
						laser_sprite.kill()

		#Alien Lasers
		if self.alien_lasers_group:
			for laser_sprite in self.alien_lasers_group:
				if pygame.sprite.spritecollide(laser_sprite, self.spaceship_group, False):
					laser_sprite.kill()
					self.lives -= 1
					if self.lives == 0:
						self.game_over()

				for obstacle in self.obstacles:
					if pygame.sprite.spritecollide(laser_sprite, obstacle.blocks_group, True):
						laser_sprite.kill()

		if self.aliens_group:
			for alien in self.aliens_group:
				for obstacle in self.obstacles:
					pygame.sprite.spritecollide(alien, obstacle.blocks_group, True)

				if pygame.sprite.spritecollide(alien, self.spaceship_group, False):
					self.game_over()

	def game_over(self) -> None:
		"""Stops the game if the user loses"""
		self.run = False

	def reset(self) -> None:
		"""Updates the playing field"""
		self.run = True
		self.lives = 3
		self.spaceship_group.sprite.reset()
		self.aliens_group.empty()
		self.alien_lasers_group.empty()
		self.create_aliens()
		self.mystery_ship_group.empty()
		self.obstacles = self.create_obstacles()
		self.score = 0

	def check_for_highscore(self) -> None:
		"""Check the record and update it if the record is broken"""
		if self.score > self.highscore:
			self.highscore = self.score

			with open("highscore.txt", "w") as file:
				file.write(str(self.highscore))

	def load_highscore(self) -> None:
		"""Load highscore from file"""
		try:
			with open("highscore.txt", "r") as file:
				self.highscore = int(file.read())
		except FileNotFoundError:
			self.highscore = 0