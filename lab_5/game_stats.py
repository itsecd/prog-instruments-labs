from game_functions import load_high_score


class GameStats():
	"""A class for tracking statistics"""

	def __init__(self, ai_settings, path_high_score):
		"""Initializes statistics"""
		self.ai_settings = ai_settings
		self.reset_stats()
		self.game_active = False
		self.high_score = load_high_score(path_high_score)

	def reset_stats(self):
		"""Initializes statistics that change during the game"""
		self.ships_left = self.ai_settings.ship_limit
		self.score = 0
		self.level = 1
 