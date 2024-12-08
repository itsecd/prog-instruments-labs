import sys
import json

from random import randint
from time import sleep

import pygame

from bullet import Bullet
from alien import Alien
from star import Star

from paths import HIGH_SCORE


def dump_high_score(stats, path_high_score):
	"""Dumps a record score"""
	with open(path_high_score, 'w') as f_name:
		json.dump(stats.high_score, f_name)


def load_high_score(path_high_score):
	"""Loads a record score"""
	try:
		with open(path_high_score) as f_name:
			high_score = json.load(f_name)
	except FileNotFoundError:
		return 0
	else:
		return high_score


def check_keydown_events(event, ship, bullets, screen, ai_settings, stats):
	"""Responds to keystrokes"""
	if event.key == pygame.K_RIGHT:
		ship.moving_right = True
	elif event.key == pygame.K_LEFT:
		ship.moving_left = True
	elif event.key == pygame.K_UP:
		ship.moving_up = True	
	elif event.key == pygame.K_DOWN:
		ship.moving_down = True
	elif event.key == pygame.K_SPACE:
		fire_bullet(ai_settings,screen, ship, bullets)
	elif event.key == pygame.K_ESCAPE:
		dump_high_score(stats, HIGH_SCORE)
		sys.exit()


def fire_bullet(ai_settings,screen, ship, bullets):
	"""Fires a bullet if the maximum has not yet been reached"""
	if(len(bullets) < ai_settings.bullets_allowed):
			new_bullet = Bullet(ai_settings, screen, ship)
			bullets.add(new_bullet)


def check_keyup_events(event, ship):
	"""Responds to keystrokes"""
	if event.key == pygame.K_RIGHT:
		ship.moving_right = False
	elif event.key == pygame.K_LEFT:
		ship.moving_left = False
	elif event.key == pygame.K_UP:
		ship.moving_up = False
	elif event.key == pygame.K_DOWN:
		ship.moving_down = False


def check_events(ai_settings, screen, ship, bullets, stats, play_button,
				 	 aliens, scoreboard):
	"""Handles keystrokes and mouse events"""
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			return False
		elif event.type == pygame.KEYDOWN:
			check_keydown_events(event, ship, bullets, screen, ai_settings, stats)	
		elif event.type == pygame.KEYUP:
			check_keyup_events(event, ship)
		elif event.type == pygame.MOUSEBUTTONDOWN:
			mouse_x, mouse_y = pygame.mouse.get_pos()
			check_play_button(mouse_x, mouse_y, stats, play_button, bullets, aliens,
					 			 ai_settings, screen, ship, scoreboard)


def check_play_button(mouse_x, mouse_y, stats, play_button, bullets, aliens,
						 ai_settings, screen, ship, scoreboard):
	"""Starts a new game when the Play button is pressed."""
	button_clicked = play_button.rect.collidepoint(mouse_x,mouse_y)
	if button_clicked and not stats.game_active:
		pygame.mouse.set_visible(False)
		ai_settings.initialize_dyn_settings()
		stats.reset_stats()
		stats.game_active = True
		scoreboard.prep_score()
		scoreboard.prep_level()
		scoreboard.prep_high_score()
		scoreboard.prep_ships()
		aliens.empty()
		bullets.empty()
		create_fleet(screen, ai_settings, aliens, ship)
		ship.center_ship()

			
def update_screen(ai_settings, screen, ship, bullets, aliens, stars, stats, 
				  play_button, scoreboard):
	"""Updates the images on the screen and displays a new screen"""
	screen.fill(ai_settings.screen_color)
	stars.draw(screen)
	for bullet in bullets.sprites():
		bullet.draw_bullet()
	ship.blitme()
	aliens.draw(screen)
	scoreboard.show_score()
	if not stats.game_active:
		play_button.draw_button()
	pygame.display.flip()


def check_high_score(stats, scoreboard):
	"""Checks if a new record has appeared"""
	if stats.score > stats.high_score:
		stats.high_score = stats.score
	scoreboard.prep_high_score()


def check_bullets_aliens_collisions(bullets, aliens, stats, ai_settings, 
										scoreboard, screen, ship):
	"""Handling bullet collisions with aliens"""
	collisions = pygame.sprite.groupcollide(bullets, aliens, True, True)
	if collisions:
		for aliens in collisions.values():
			stats.score += ai_settings.alien_points * len(aliens)
	scoreboard.prep_score()
	check_high_score(stats, scoreboard)
	if len(aliens) == 0:
		bullets.empty()
		ai_settings.increase_speed()
		stats.level += 1
		scoreboard.prep_level()
		create_fleet(screen, ai_settings, aliens, ship)


def update_bullets(bullets, aliens, screen, ai_settings, ship, scoreboard,
				   	 stats):
	"""Updates bullet positions and destroys old bullets"""
	bullets.update()
	for bullet in bullets.copy():
			if(bullet.rect.bottom <= 0):
				bullets.remove(bullet)				
	check_bullets_aliens_collisions(bullets, aliens, stats, ai_settings, scoreboard,
								 		 screen, ship)


def get_number_aliens_x(ai_settings, alien_width):
	"""Calculates the number of aliens in a row"""
	avaliable_space_x = ai_settings.screen_width - 2 * alien_width
	number_aliens_x = int(avaliable_space_x / (2 * alien_width))
	return number_aliens_x


def get_number_rows(ai_settings, alien_height, ship_height):
	"""Calculates the number of rows that fit on the screen"""
	avaliable_space_y = ai_settings.screen_height - ship_height - 3 * alien_height
	number_rows = int(avaliable_space_y / alien_height)
	return number_rows


def create_alien(alien_number, screen, ai_settings, aliens, row_number):
	"""Creates an alien and places it in a row"""
	alien = Alien(screen, ai_settings)
	alien_width = alien.rect.width
	alien_height = alien.rect.height
	alien.y = alien_height + alien_height * row_number
	alien.x = alien_width + 2 * alien_width * alien_number
	alien.rect.x = alien.x 
	alien.rect.y = alien.y
	aliens.add(alien)


def create_fleet(screen, ai_settings, aliens, ship):
	"""Creates an alien fleet"""
	alien = Alien(screen, ai_settings)
	number_aliens_x = get_number_aliens_x(ai_settings, alien.rect.width)
	number_rows = get_number_rows(ai_settings, alien.rect.height, ship.rect.height)
	for row_number in range(number_rows):
		for alien_number in range(number_aliens_x):
			create_alien(alien_number, screen, ai_settings, aliens, row_number)


def get_number_stars_x(ai_settings, star_width):
	"""Calculates the number of stars in a row"""
	avaliable_space_x = ai_settings.screen_width - 2 * star_width
	number_stars =int(avaliable_space_x / (2 * star_width))
	return number_stars


def get_stars_rows(ai_settings, star_height):
	"""Calculates the number of rows that fit on the screen"""
	avaliable_space_y = ai_settings.screen_height - 2 * star_height
	number_rows = int(avaliable_space_y / (2 * star_height))
	return number_rows


def create_star(star_number, row_number, stars, screen, ai_settings):
	"""Creates a star and places it in a row"""
	star = Star(screen)
	star_width = star.rect.width 
	star_height = star.rect.height
	star.x =  star_width + randint(0, ai_settings.screen_width) * star_number
	star.y =  star_height  + randint(0, ai_settings.screen_height) * row_number
	star.rect.x = star.x 
	star.rect.y = star.y 
	stars.add(star)


def create_stars(screen, ai_settings, stars):
	"""Creates all stars"""
	star = Star(screen)
	number_stars = get_number_stars_x(ai_settings, star.rect.width)
	number_rows = get_stars_rows(ai_settings, star.rect.height) 
	for row_number in range(number_rows):
		for star_number in range(number_stars):
			create_star(star_number, row_number, stars, screen, ai_settings)


def check_fleet_edges(aliens, ai_settings):
	"""Reacts to the alien reaching the edge of the screen"""
	for alien in aliens.sprites():
		if alien.check_edges():
			change_fleet_direction(aliens, ai_settings)
			break


def change_fleet_direction(aliens, ai_settings):
	"""Lowers the entire fleet and changes the direction of the fleet"""
	for alien in aliens.sprites():
		alien.rect.y += ai_settings.fleet_drop_speed
	ai_settings.fleet_direction *= -1


def ship_hit(aliens, ai_settings, ship, stats, screen, bullets, scoreboard):
	"""Handles the collision of a ship with an alien"""
	if stats.ships_left > 0:
		stats.ships_left -= 1
		scoreboard.prep_ships()
		aliens.empty()
		bullets.empty()
		create_fleet(screen, ai_settings, aliens, ship)
		ship.center_ship()
		sleep(1)
	else:
		stats.game_active = False
		pygame.mouse.set_visible(True)


def update_aliens(aliens, ai_settings, ship, stats, screen, bullets,
				  	 scoreboard):
	"""Updates the positions of all aliens in the fleet"""
	check_fleet_edges(aliens, ai_settings)
	aliens.update()
	if pygame.sprite.spritecollideany(ship, aliens):
		ship_hit(aliens, ai_settings, ship, stats, screen, bullets, scoreboard)

	check_aliens_bottom(aliens, ai_settings, ship, stats, screen, bullets,
					 	 scoreboard)


def check_aliens_bottom(aliens, ai_settings, ship, stats, screen, bullets,
							 scoreboard):
	"""Checks if the aliens have reached the bottom of the screen"""
	for alien in aliens.sprites():
		if alien.rect.bottom >= screen.get_rect().bottom:
			ship_hit(aliens, ai_settings, ship, stats, screen, bullets, scoreboard)
			break
 