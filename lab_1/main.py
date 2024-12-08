from typing import List, Optional, Tuple
from time import time
from random import randint, uniform

import pygame

from enemy import Enemy
from high_scores_screen import screen_scores
from functions import animate_gif, play_sound
from register import (BESTSCORE, NAMEID, PLAYING, SCORE, height, pickle,
                      scores, screen, width)
from variables import (BESTSCOREBEATEN, BESTSCOREPATH, DAMAGE, DAMAGEPATH,
                       DUCKSTATE, HEALING, HEALINGPATH, JUMPING, JUMPPATH,
                       LOST, LOSTPATH, SCORE1000PATH, SCOREPATH,
                       SONICSTANDINGSTATE, SONICSTATE, TIMEJUMP,
                       best_score_rect, best_score_surface, best_score_time,
                       cloud_2_rect, cloud_rect, effect_time, end_font,
                       end_rect, end_surface, end_time, enemies,
                       enemy_bird_surface, enemy_spike_surface, game_over_rect,
                       game_over_surface, grass_2_rect, grass_3_rect,
                       grass_rect, grass_surface, heart_rect, heart_surface,
                       last_score_rect, last_score_surface, palm_2_rect,
                       palm_rect, pseudo_rect, pseudo_surface, restart_rect,
                       restart_surface, rock_surface, score_font,
                       score_live_font, scores_screen_rect,
                       scores_screen_surface, sonic_1_rect, sonic_jump_rect,
                       sonic_jump_surface, sonic_rect, start_jump, states_duck,
                       states_sonic, time_gif, time_gif_duck, time_score_sound,
                       time_spawn, timer)


def handle_events() -> None:
    """
    Handles player input and updates game state accordingly.
    """
    global PLAYING, JUMPING, LOST, SCORE, time_score_sound
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            PLAYING = False
        elif event.type == pygame.KEYDOWN:
            handle_keydown(event)


def handle_keydown(event: pygame.event.Event) -> None:
    """
    Processes key press events.
    """
    global JUMPING, LOST, start_jump, SCORE
    if event.key == pygame.K_SPACE:
        if LOST:
            reset_game()
        elif sonic_jump_rect.on_floor():
            start_jump = time()
            JUMPING = True
            play_sound(JUMPPATH, 0.02)
            sonic_jump_rect.change_speed((0, 1300 - ACCELERATION / 2.5))


def reset_game() -> None:
    """
    Resets the game to its initial state after the player loses.
    """
    global LOST, SCORE, start_jump, enemies, end_time, scores
    LOST = False
    SCORE = 0
    start_jump = time()
    enemies.clear()
    sonic_1_rect.health = 3
    play_sound(LOSTPATH, 0.06)
    end_time = time()
    scores[NAMEID] = BESTSCORE
    with open("best_score.pickle", "wb") as f:
        pickle.dump(scores, f)


def spawn_enemies() -> None:
    """
    Spawns enemies based on game logic and player state.
    """
    global time_spawn
    if time() >= time_spawn + calculate_enemy_delay():
        enemies.append(generate_enemy())
        time_spawn = time()


def calculate_enemy_delay() -> float:
    """
    Calculates the delay before spawning the next enemy.
    """
    mobs_speed = 850 + ACCELERATION
    return 150 * 4.8 / mobs_speed + uniform(-0.05, 0.7)


def generate_enemy() -> Enemy:
    """
    Generates an enemy based on random conditions and player state.
    """
    rand = randint(1, 10)
    if rand <= 7:
        return Enemy(rock_surface.get_rect(topleft=(width, height - 200)), rock_surface, "littleMob")
    return Enemy(enemy_bird_surface.get_rect(topleft=(width, 300)), enemy_bird_surface, "flyingMob")


def update_game_state() -> None:
    """
    Updates the game state, including score, animations, and collisions.
    """
    update_score()
    handle_collisions()


def update_score() -> None:
    """
    Updates the score and checks for new high scores.
    """
    global SCORE, BESTSCORE, BESTSCOREBEATEN
    if not LOST:
        SCORE = int((time() - score_timer) * 10)
        if SCORE > BESTSCORE:
            BESTSCORE = SCORE
            BESTSCOREBEATEN = True


def handle_collisions() -> None:
    """
    Processes collisions between the player and enemies or objects.
    """
    global DAMAGE, HEALING
    for enemy in enemies:
        if enemy.rect.colliderect(sonic_jump_rect.rect):
            if enemy.category == "heart":
                heal_player()
            else:
                damage_player()


def heal_player() -> None:
    """
    Heals the player when colliding with a heart.
    """
    global HEALING
    if sonic_1_rect.health < 6:
        sonic_1_rect.health += 1
    HEALING = True


def damage_player() -> None:
    """
    Reduces the player's health and triggers damage effect.
    """
    global DAMAGE
    sonic_1_rect.health -= 1
    DAMAGE = True


def render_screen() -> None:
    """
    Renders all game elements on the screen.
    """
    screen.fill((135, 206, 235))  # Clear screen
    render_background()
    render_enemies()
    render_player()
    pygame.display.flip()


def render_background() -> None:
    """
    Renders the background and other decorative elements.
    """
    grass_rect.animate(850 + ACCELERATION, 0, timer.tick(200) / 1000, screen)
    cloud_rect.animate(620, 0, timer.tick(200) / 1000, screen)


def render_enemies() -> None:
    """
    Renders all enemies currently on the screen.
    """
    for enemy in enemies:
        enemy.display(screen)


def render_player() -> None:
    """
    Renders the player character.
    """
    if JUMPING:
        screen.blit(sonic_jump_surface, sonic_jump_rect.rect)
    else:
        screen.blit(states_sonic[0][SONICSTATE], (100, height - 200 - 144))


# Main game loop
while PLAYING:
    handle_events()
    spawn_enemies()
    update_game_state()
    render_screen()
pygame.quit()
