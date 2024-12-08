import pygame.font

from pygame.sprite import Group

from ship import Ship


class Scoreboard():
	"""A class for displaying game information"""

	def __init__(self, screen, ai_settings, stats):
		"""Initializes scoring attributes"""
		self.screen = screen
		self.ai_settings = ai_settings
		self.stats = stats
		self.screen_rect = self.screen.get_rect()
		self.text_color = (30, 30, 30)
		self.font = pygame.font.SysFont(None, 48)
		self.prep_score()
		self.prep_high_score()
		self.prep_level()
		self.prep_ships()

	def prep_score(self):
		"""Converts the current account into a graphic image"""
		rounded_score = round(self.stats.score, -1)
		score_str = "{:,}".format(rounded_score)
		self.score_image = self.font.render(score_str, True, self.text_color,
									  		 self.ai_settings.screen_color)
		self.score_rect = self.score_image.get_rect()
		self.score_rect.right = self.screen_rect.right - 20
		self.score_rect.top = 20

	def prep_high_score(self):
		"""Converts the record score into a graphic image"""
		rounded_high_score = round(self.stats.high_score, -1)
		self.stats.high_score = rounded_high_score
		high_score_str = "{:,}".format(rounded_high_score)
		self.high_score_image = self.font.render(high_score_str, True, self.text_color,
										   			self.ai_settings.screen_color)
		self.high_score_rect = self.high_score_image.get_rect()
		self.high_score_rect.center = self.screen_rect.center
		self.high_score_rect.top = 20

	def prep_level(self):
		"""Converts the level into a graphic image"""
		self.level_image = self.font.render(str(self.stats.level), True,
									  		 self.text_color, self.ai_settings.screen_color)
		self.level_rect = self.level_image.get_rect()
		self.level_rect.right = self.score_rect.right
		self.level_rect.top = self.score_rect.bottom + 10

	def prep_ships(self):
		"""Reports the number of remaining ships"""
		self.ships = Group()
		for ship_number in range(self.stats.ships_left):
			ship = Ship(self.screen, self.ai_settings)
			ship.rect.x = 10 + ship.rect.width * ship_number
			ship.rect.y = 10
			self.ships.add(ship)

	def show_score(self):
		"""Converts the score into a graphic image"""
		self.screen.blit(self.score_image, self.score_rect)
		self.screen.blit(self.high_score_image, self.high_score_rect)
		self.screen.blit(self.level_image, self.level_rect)
		self.ships.draw(self.screen)
 