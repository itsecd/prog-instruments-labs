class Settings():
	"""A class for storing all game settings"""

	def __init__(self):
		"""Initializes the game settings"""
		self.screen_width = 1280
		self.screen_height = 720
		self.screen_color = (42, 65, 127)
		self.ship_limit = 3
		self.bullet_width = 3
		self.bullet_height = 15
		self.bullet_color = (203, 13, 26)
		self.bullets_allowed = 3
		self.fleet_drop_speed = 7
		self.speedup_scale = 1.2
		self.score_scale = 1.5

	def initialize_dyn_settings(self):
		"""Initializes settings that change during the game"""
		self.ship_speed_factor = 1.5
		self.bullet_speed_factor = 3
		self.alien_speed_factor = 1
		self.fleet_direction = 1
		self.alien_points = 50

	def increase_speed(self):
		"""Increases the speed settings"""
		self.ship_speed_factor *= self.speedup_scale
		self.bullet_speed_factor *= self.speedup_scale
		self.alien_speed_factor *= self.speedup_scale
		self.alien_points = int(self.alien_points * self.score_scale)
 