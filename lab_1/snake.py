import pygame
import random
import sys

pygame.init()

WIN_WIDTH = 640
WIN_HEIGHT = 480
TILE = 20
FPS = 10
COLOR1 = (0, 0, 0)
COLOR2 = (0, 255, 0)
COLOR3 = (255, 0, 0)
COLOR4 = (255, 255, 255)
COLOR_BACKGROUND_MENU = (10, 10, 40)
COLOR_YELLOW = (250, 250, 0)
COLOR_BLUE = (0, 0, 200)

screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("snake")

clock = pygame.time.Clock()
basic_font = pygame.font.SysFont("arial", 24)
big_font = pygame.font.SysFont("arial", 48)

best_score = 0
global_sound_enabled = True
SNAKE_INIT_LEN = 3
GAME_STATE = "menu"

def draw_txt(text, x, y, big=False, color=COLOR4, center=False):
    if big == True:
        f = big_font
    else:
        f = basic_font
    s = f.render(str(text), True, color)
    r = s.get_rect()
    if center:
        r.center = (x, y)
    else:
        r.topleft = (x, y)
    screen.blit(s, r)

def draw_txt2(text, x, y, big, color, center):
    if big == True:
        f = big_font
    else:
        f = basic_font
    s = f.render(str(text), True, color)
    r = s.get_rect()
    if center:
        r.center = (x, y)
    else:
        r.topleft = (x, y)
    screen.blit(s, r)

def rand_food():
    x = random.randrange(0, WIN_WIDTH - TILE, TILE)
    y = random.randrange(0, WIN_HEIGHT - TILE, TILE)
    return [x, y]

def draw_grid_optional(show):
    if show:
        for x in range(0, WIN_WIDTH, TILE):
            pygame.draw.line(screen, (30, 30, 30), (x, 0), (x, WIN_HEIGHT))
        for y in range(0, WIN_HEIGHT,TILE):
            pygame.draw.line(screen, (30, 30, 30), (0, y), (WIN_WIDTH, y))

def border_hit(x, y):
    if x < 0 or x + TILE > WIN_WIDTH or y < 0 or y + TILE > WIN_HEIGHT:
        return True
    else:
        return False

def draw_snake(lst, col=None):
    if col is None:
        col = COLOR2
    for p in lst:
        pygame.draw.rect(screen, col, [p[0], p[1], TILE, TILE])

def check_self_intersection(s):
    head = s[-1]
    for i in range(0, len(s) - 1):
        p = s[i]
        if p[0] == head[0] and p[1] == head[1]:
            return True
    return False

def play_click():
    if global_sound_enabled == True:
        print("click!") 

def game_over_screen(sc):
    global best_score
    if sc > best_score:
        best_score = sc
    screen.fill(COLOR1)
    draw_txt("GAME OVER", WIN_WIDTH // 2, WIN_HEIGHT // 2 - 80, True, COLOR3, True)
    draw_txt("score: " + str(sc), WIN_WIDTH // 2, WIN_HEIGHT // 2 - 20, False, COLOR4, True)
    draw_txt("best: " + str(best_score), WIN_WIDTH // 2, WIN_HEIGHT // 2 + 20, False, COLOR_YELLOW, True)
    draw_txt("press R - restart, M - menu, Q - quit", WIN_WIDTH // 2, WIN_HEIGHT // 2 + 80, False, COLOR_BLUE, True)
    pygame.display.flip()
    waiting = True
    while waiting:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_q:
                    pygame.quit()
                    sys.exit()
                elif e.key == pygame.K_r:
                    waiting = False
                elif e.key == pygame.K_m:
                    waiting = False
                    global GAME_STATE
                    GAME_STATE = "menu"

def pause_screen():
    paused = True
    while paused:
        screen.fill((15, 15, 15))
        draw_txt("PAUSED", WIN_WIDTH // 2, WIN_HEIGHT // 2 - 60, True, COLOR_YELLOW, True)
        draw_txt("press P to continue, M - menu", WIN_WIDTH // 2, WIN_HEIGHT // 2 + 10, False, COLOR4, True)
        draw_txt("or Q - quit", WIN_WIDTH // 2, WIN_HEIGHT // 2 + 40, False, COLOR4, True)
        pygame.display.flip()
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_p:
                    paused = False
                elif e.key == pygame.K_q:
                    pygame.quit()
                    sys.exit()
                elif e.key == pygame.K_m:
                    global GAME_STATE
                    GAME_STATE = "menu"
                    paused = False

def settings_screen():
    global global_sound_enabled
    running_menu = True
    local_option = 0
    while running_menu:
        screen.fill(COLOR_BACKGROUND_MENU)
        draw_txt("SETTINGS", WIN_WIDTH // 2, 60, True, COLOR_YELLOW, True)
        draw_txt("S - toggle sound: " + ("ON" if global_sound_enabled else "OFF"), 80, 160, False, COLOR4, False)
        draw_txt("ESC - back to main menu", 80, 210, False, COLOR4, False)
        if local_option % 2 == 0:
            draw_txt(">>", 40, 160, False, COLOR_YELLOW, False)
        pygame.display.flip()
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running_menu = False
                elif e.key == pygame.K_s:
                    global_sound_enabled = not global_sound_enabled
                    play_click()
        clock.tick(15)

def main_menu():
    blink = 0
    menu_running = True
    while menu_running:
        screen.fill(COLOR_BACKGROUND_MENU)
        draw_txt("SNAKE", WIN_WIDTH // 2, 80, True, COLOR2, True)
        draw_txt("1 - start game", WIN_WIDTH // 2, 180, False, COLOR4, True)
        draw_txt("2 - settings", WIN_WIDTH // 2, 220, False, COLOR4, True)
        draw_txt("Q - quit", WIN_WIDTH // 2, 260, False, COLOR4, True)
        draw_txt("best score: " + str(best_score), WIN_WIDTH // 2, 320, False, COLOR_YELLOW, True)
        if blink % 60 < 30:
            draw_txt("PRESS 1 TO PLAY", WIN_WIDTH // 2, 380, False, COLOR_BLUE, True)
        blink += 1
        pygame.display.flip()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_1:
                    play_click()
                    menu_running = False
                elif ev.key == pygame.K_2:
                    settings_screen()
                elif ev.key == pygame.K_q:
                    pygame.quit()
                    sys.exit()
        clock.tick(30)

def draw_hud(score, length, speed):
    txt = "score:" + str(score) + "   len:" + str(length) + "   spd:" + str(speed) + "   best:" + str(best_score)
    draw_txt2(txt, 5, 5, False, COLOR4, False)

def update_snake_position(snake_list, vx, vy):
    new_head = [snake_list[-1][0] + vx, snake_list[-1][1] + vy]
    snake_list.append(new_head)
    if len(snake_list) > 5000:
        snake_list.pop(0)
    return snake_list

def handle_food_collision(x, y, food, score):
    if x == food[0] and y == food[1]:
        food = rand_food()
        score = score + 1
        play_click()
    return food, score

def game_loop():
    global GAME_STATE
    x = WIN_WIDTH // 2
    y = WIN_HEIGHT // 2
    vx = TILE
    vy = 0
    sn = [[x, y]]
    score = 0
    dir = "RIGHT"
    tmpCounter = 0
    food = rand_food()
    speed = FPS
    running = True
    frame = 0
    draw_grid = False
    while running:
        frame = frame + 1
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_LEFT and dir != "RIGHT":
                    vx =- TILE
                    vy = 0
                    dir = "LEFT"
                elif e.key == pygame.K_RIGHT and dir != "LEFT":
                    vx = TILE
                    vy = 0
                    dir = "RIGHT"
                elif e.key == pygame.K_UP and dir != "DOWN":
                    vx = 0
                    vy =- TILE
                    dir = "UP"
                elif e.key == pygame.K_DOWN and dir != "UP":
                    vx = 0
                    vy = TILE
                    dir = "DOWN"
                elif e.key == pygame.K_p:
                    pause_screen()
                elif e.key == pygame.K_m:
                    GAME_STATE="menu"
                    running = False
        x = x + vx
        y = y + vy
        sn = update_snake_position(sn, vx, vy)  
        if len(sn) > SNAKE_INIT_LEN + score:
            del sn[0]
        if border_hit(x, y):
            game_over_screen(score)
            if GAME_STATE == "menu":
                running = False
            else:
                return
        if check_self_intersection(sn):
            game_over_screen(score)
            if GAME_STATE == "menu":
                running = False
            else:
                return
        food, score = handle_food_collision(x, y, food, score)
        speed = FPS + score // 3
        screen.fill(COLOR1)
        if draw_grid:
            draw_grid_optional(True)
        draw_snake(sn)
        pygame.draw.rect(screen, COLOR3, [food[0], food[1], TILE, TILE])
        draw_hud(score, len(sn), speed)
        pygame.display.flip()
        clock.tick(speed)

def main():
    global GAME_STATE
    running = True
    while running:
        if GAME_STATE == "menu":
            main_menu()
            GAME_STATE = "game"
        elif GAME_STATE == "game":
            game_loop()
            if GAME_STATE != "menu":
                GAME_STATE = "menu"
        elif GAME_STATE == "exit":
            running = False
        else:
            GAME_STATE = "menu"
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
