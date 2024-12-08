import pygame

from pygame.sprite import Group

import game_functions as gf

from settings import Settings
from ship import Ship
from game_stats import GameStats
from button import Button
from scoreboard import Scoreboard
from paths import HIGH_SCORE


def run_game():
	"""Initializes the game and creates a screen object"""
	pygame.init()
	ai_settings = Settings()
	screen = pygame.display.set_mode((ai_settings.screen_width,
								   		 ai_settings.screen_height))
	pygame.display.set_caption("Alien Invasion")
	play_button = Button(screen, "Play")
	stats = GameStats(ai_settings, HIGH_SCORE)
	scoreboard = Scoreboard(screen, ai_settings, stats)
	ship = Ship(screen, ai_settings)
	print(ship.center)
	bullets = Group()
	aliens = Group()
	stars = Group()
	gf.create_stars(screen, ai_settings, stars)
	gf.create_fleet(screen, ai_settings, aliens, ship)
	while True:
		if gf.check_events(ai_settings, screen, ship, bullets, stats, play_button,
				  		 aliens, scoreboard) == False:
			break
		if stats.game_active:
			ship.update()
			gf.update_bullets(bullets, aliens, screen, ai_settings, ship, scoreboard, 
					 			stats)
			gf.update_aliens(aliens, ai_settings, ship, stats, screen, bullets, scoreboard)
		gf.update_screen(ai_settings, screen, ship, bullets, aliens, stars, stats,
				   			 play_button, scoreboard)
	pygame.quit()


run_game()
 