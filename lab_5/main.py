import pygame

from player import Player
from enemy import Enemy
from config import WIDTH, HEIGHT, BLACK, WHITE, RED, ENEMY_COUNT

pygame.init()


def main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("OOOh my rect go test")
    clock = pygame.time.Clock()

    player = Player()
    enemy_list = [Enemy() for enemy in range(ENEMY_COUNT)]

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

        for enemy in enemy_list:
            if player.rect.colliderect(enemy.rect):
                enemy_list.remove(enemy)

        screen.fill(WHITE)
        pygame.draw.rect(screen, BLACK, player.rect)

        for enemy in enemy_list:
            pygame.draw.rect(screen, RED, enemy.rect)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
