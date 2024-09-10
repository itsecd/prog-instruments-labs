from game_functions import load_high_score
class GameStats():
	def __init__(self, ai_settings):
		self.ai_settings = ai_settings
		self.reset_stats()
		self.game_active = False
		self.high_score = load_high_score()

	def reset_stats(self):
		self.ships_left = self.ai_settings.ship_limit
		self.score = 0
		self.level = 1


