import pygame

from game import Player
from config import WIDTH, HEIGHT, BLACK, WHITE

pygame.init()


def main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("OOOh my rect go test")
    clock = pygame.time.Clock()

    player = Player()

    cycle_run = True
    while cycle_run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cycle_run = False

        keys = pygame.key.get_pressed()
        move_x, move_y = 0, 0
        if keys[pygame.K_LEFT]:
            move_x = -5
        if keys[pygame.K_RIGHT]:
            move_x = 5
        if keys[pygame.K_UP]:
            move_y = -5
        if keys[pygame.K_DOWN]:
            move_y = 5

        player.move(move_x, move_y)

        screen.fill(WHITE)
        pygame.draw.rect(screen, BLACK, player.rect)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
