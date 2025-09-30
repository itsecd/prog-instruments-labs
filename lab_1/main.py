import sys
import pygame

from Tetris import (
    make_SHAPE_CONFIG, new_game, disp_start,
    key_down, disp_score, process_timer
)

def main():
    make_SHAPE_CONFIG()
    new_game()
    disp_start()

    # pygame.time.set_timer(pygame.USEREVENT, 500)

    # main game loop
    while True:

        # Get Keyboard
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                key_down(event)
                disp_score()
                pygame.display.flip()
            elif event.type == pygame.USEREVENT:
                process_timer(event)
                disp_score()
                pygame.display.flip()

if __name__ == '__main__':
    main()
