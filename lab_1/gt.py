import pygame

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

p1_health = 10
p2_health = 10

LEFT_SPEED = (-6, 0)
RIGHT_SPEED = (6, 0)
DOWN_SPEED = (0, 6)
UP_SPEED = (0, -6)

'''
Player1 uses the skull and the fireball(bullet) stays at the right side
Player2 uses the togepi and the cloud(bullet2) stays at teh left side
'''


class Player(pygame.sprite.Sprite):
    change_x = 0
    change_y = 0
    walls = None

    # Constructor
    def __init__(self, x, y):
        super().__init__()

        # self.image = pygame.Surface([20, 20])
        # self.image.fill(WHITE)
        self.image = pygame.image.load("skull.jpeg").convert()

        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

    def changespeed(self, x, y):
        self.change_x += x
        self.change_y += y

    def update(self):
        self.rect.x += self.change_x

        block_hit_list = pygame.sprite.spritecollide(self, self.walls, False)
        for block in block_hit_list:
            if self.change_x > 0:
                self.rect.right = block.rect.left

            elif self.change_x < 0:
                self.rect.left = block.rect.right

        self.rect.y += self.change_y

        block_hit_list = pygame.sprite.spritecollide(self, self.walls, False)
        for block in block_hit_list:
            if self.change_y > 0:
                self.rect.bottom = block.rect.top
            elif self.change_y < 0:
                self.rect.top = block.rect.bottom


class Player2(pygame.sprite.Sprite):
    change_x = 0
    change_y = 0
    walls = None

    # Constructor
    def __init__(self, x, y):
        super().__init__()

        # self.image = pygame.Surface([20, 20])
        # self.image.fill(WHITE)
        self.image = pygame.image.load("rsz_togepi.jpg").convert()

        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

    def changespeed(self, x, y):
        self.change_x += x
        self.change_y += y

    def update(self):
        self.rect.x += self.change_x

        block_hit_list = pygame.sprite.spritecollide(self, self.walls, False)
        for block in block_hit_list:
            if self.change_x > 0:
                self.rect.right = block.rect.left

            elif self.change_x < 0:
                self.rect.left = block.rect.right

        self.rect.y += self.change_y

        block_hit_list = pygame.sprite.spritecollide(self, self.walls, False)
        for block in block_hit_list:
            if self.change_y > 0:
                self.rect.bottom = block.rect.top
            elif self.change_y < 0:
                self.rect.top = block.rect.bottom


class Cloud(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()

        self.image = pygame.image.load("cloud.png").convert()

        self.rect = self.image.get_rect()

    def update(self):
        self.rect.x += 9


class Fireball(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()

        self.image = pygame.image.load("fireball.jpg").convert()

        self.rect = self.image.get_rect()

    def update(self):
        self.rect.x -= 9


class Wall(pygame.sprite.Sprite):
    def __init__(self, x, y, width, height):
        super().__init__()
        self.image = pygame.Surface([width, height])
        self.image.fill(BLUE)

        self.rect = self.image.get_rect()
        self.rect.y = y
        self.rect.x = x


pygame.init()

screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])
pygame.display.set_caption("Game Thingy")
all_sprite_list = pygame.sprite.Group()
player1_list = pygame.sprite.Group()
player2_list = pygame.sprite.Group()

wall_list = pygame.sprite.Group()

wall_1 = Wall(0, 0, 10, 600)
wall_list.add(wall_1)
all_sprite_list.add(wall_1)

wall_2 = Wall(790, 0, 10, 600)
wall_list.add(wall_2)
all_sprite_list.add(wall_2)

wall_3 = Wall(400, 0, 10, 600)
wall_list.add(wall_3)
all_sprite_list.add(wall_3)

bullet_list = pygame.sprite.Group()
bullet2_list = pygame.sprite.Group()

player1 = Player(650, 50)
player1.walls = wall_list

all_sprite_list.add(player1)
player1_list.add(player1)

player2 = Player2(50, 50)
player2.walls = wall_list
player2_list.add(player2)

all_sprite_list.add(player2)


font = pygame.font.SysFont('Arial', 25, False, False)
'''
player1health = font.render("Health: " + str(p1_health), True, WHITE)

player2health = font.render("Health: " + str(p2_health), True, WHITE)
'''

text2 = font.render("Game Over P2 Wins", True, WHITE)
text2_rect = text2.get_rect()
text2_x = screen.get_width() / 2 - text2_rect.width / 2
text2_y = screen.get_height() / 2 - text2_rect.height / 2

text1 = font.render("Game Over P1 Wins", True, WHITE)
text1_rect = text1.get_rect()
text1_x = screen.get_width() / 2 - text1_rect.width / 2
text1_y = screen.get_height() / 2 - text1_rect.height / 2

clock = pygame.time.Clock()

done = False
gameover = False
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        if event.type == pygame.KEYDOWN and gameover is False:
            if event.key == pygame.K_LEFT:
                player1.changespeed(*LEFT_SPEED)
            elif event.key == pygame.K_RIGHT:
                player1.changespeed(*RIGHT_SPEED)
            elif event.key == pygame.K_DOWN:
                player1.changespeed(*DOWN_SPEED)
            elif event.key == pygame.K_UP:
                player1.changespeed(*UP_SPEED)

            elif event.key == pygame.K_a:
                player2.changespeed(*LEFT_SPEED)
            elif event.key == pygame.K_d:
                player2.changespeed(*RIGHT_SPEED)
            elif event.key == pygame.K_s:
                player2.changespeed(*DOWN_SPEED)
            elif event.key == pygame.K_w:
                player2.changespeed(*UP_SPEED)

            elif event.key == pygame.K_SLASH:
                bullet = Fireball()
                bullet.rect.x = player1.rect.x
                bullet.rect.y = player1.rect.y

                all_sprite_list.add(bullet)
                bullet_list.add(bullet)

            elif event.key == pygame.K_SPACE:
                bullet2 = Cloud()
                bullet2.rect.x = player2.rect.x
                bullet2.rect.y = player2.rect.y

                all_sprite_list.add(bullet2)
                bullet2_list.add(bullet2)

        elif event.type == pygame.KEYUP and gameover is False:
            if event.key == pygame.K_LEFT:
                player1.changespeed(*LEFT_SPEED)
            elif event.key == pygame.K_RIGHT:
                player1.changespeed(*RIGHT_SPEED)
            elif event.key == pygame.K_DOWN:
                player1.changespeed(*DOWN_SPEED)
            elif event.key == pygame.K_UP:
                player1.changespeed(*UP_SPEED)

            elif event.key == pygame.K_a:
                player2.changespeed(*LEFT_SPEED)
            elif event.key == pygame.K_d:
                player2.changespeed(*RIGHT_SPEED)
            elif event.key == pygame.K_s:
                player2.changespeed(*DOWN_SPEED)
            elif event.key == pygame.K_w:
                player2.changespeed(*UP_SPEED)

    all_sprite_list.update()

    for bullet in bullet_list:
        block_hit_list = pygame.sprite.spritecollide(
            bullet,
            player2_list,
            False
        )

        for block in block_hit_list:
            bullet_list.remove(bullet)
            all_sprite_list.remove(bullet)
            p2_health -= 1

        if bullet.rect.x < -10:
            bullet_list.remove(bullet)
            all_sprite_list.remove(bullet)

    for bullet2 in bullet2_list:
        block_hit_list2 = pygame.sprite.spritecollide(
            bullet2,
            player1_list,
            False
        )

        for block in block_hit_list2:
            bullet2_list.remove(bullet2)
            all_sprite_list.remove(bullet2)
            p1_health -= 1

        if bullet2.rect.x > 810:
            bullet2_list.remove(bullet2)
            all_sprite_list.remove(bullet2)

    screen.fill(BLACK)
    player1health = font.render(
        "Player 1 Health: " + str(p1_health),
        True,
        WHITE
    )

    player2health = font.render(
        "Player 2 Health: " + str(p2_health),
        True,
        WHITE
    )
    screen.blit(player1health, [550, 10])
    screen.blit(player2health, [20, 10])

    all_sprite_list.draw(screen)

    if p1_health <= 0:
        screen.blit(text2, [text2_x, text2_y])
        gameover = True
    elif p2_health <= 0:
        screen.blit(text1, [text1_x, text1_y])
        gameover = True

    pygame.display.flip()

    clock.tick(60)


pygame.quit()
