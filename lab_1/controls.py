# Импорт модулей и файлов
import random
import time
import sys

import pygame

from bullets import Bullets
from comets import Comets
from stars import Stars
from additions import AddBullets
from coin import Coins
import button
from explosion import Explosion


pygame.init()  # Инициализация pygame


def events(screen, player, bullets, comets, stats,
           score, button, stars, addbullets):
    """Обработка действий человека"""
    W, H = screen.get_size()
    pressed = pygame.mouse.get_pressed()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                player.mtop = True
            if event.key == pygame.K_s:
                player.mbottom = True
            if event.key == pygame.K_a:
                player.mleft = True
            if event.key == pygame.K_d:
                player.mright = True
            if event.key == pygame.K_SPACE:
                bullets_movement(screen, player, bullets, stats, score)
            if event.key == pygame.K_ESCAPE:
                pause(screen, stats, button, player)

        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_w:
                player.mtop = False
            if event.key == pygame.K_s:
                player.mbottom = False
            if event.key == pygame.K_a:
                player.mleft = False
            if event.key == pygame.K_d:
                player.mright = False        

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                bullets_movement(screen, player, bullets, stats, score)

        if event.type == pygame.USEREVENT:
            create_comets(screen, comets, W)

        if event.type == pygame.USEREVENT + 1:
            _stars(screen, stars, W)
            stats.score += 1

        if event.type == pygame.USEREVENT + 2:
            create_bullets(screen, addbullets, W)


def update(screen, player, bullets, comets, score, stats, stars, addbullets,
           coins, explosions):
    """Отображение и обновление движение объектов на экране """
    W, H = screen.get_size()

    FPS = 250
    clock = pygame.time.Clock()

    # Изображения
    image_coin = pygame.image.load(f"images/coin.png")
    image_bullets = pygame.image.load("images/Improvements/addbullets.png")

    screen.fill((0, 0, 0))
    score.show_score()

    for star in stars.sprites():
        star.draw()

    for coin in coins.sprites():
        coin.draw()

    for bullet in bullets.sprites():
        bullet.draw()

    for comet in comets.sprites():
        comet.draw()

    for addbullet in addbullets.sprites():
        addbullet.draw_addbullet()

    screen.blit(image_bullets, score.screen_rect.right - 40,
                score.screen_rect.bottom - 55
                )

    button.print_text(screen, "Очки:", W//1.125, 19, 38)
    button.print_text(screen, "Бутк:", W/1.12, 99, 38)

    player.draw()
    explosions.draw(screen)

    pygame.display.flip()
    clock.tick(FPS)

    # Другие проверки
    with open("files/hts.txt", "r") as file:
        hearts = int(file.read())
        limit_hearts = hearts - 25

    with open("files/pl_hd.txt", "r") as file:
        hard = int(file.read())

    if stats.score % 1000 == 0 and stats.score > 0:
        # Пополнение здоровья
        if stats.player_left <= limit_hearts:
            if stats.player_left <= 100 and score.hard <= 150:
                stats.player_left += 3
                score.hard += 4.5
        else:
            stats.player_left = hearts
            score.hard = hard

        # Звук пополнения здоровья
        hards = pygame.mixer.Sound("sounds/hearts.mp3")
        hards.set_volume(stats.objectsvolume)
        hards.play()

    if stats.player_left <= 30:
        # Проверка цвета жизней
        score.color = (255, 0, 0)
    elif stats.player_left <= 60 and stats.player_left > 30:
        score.color = (255, 165, 0)
    else:
        score.color = (0, 255, 0)


def update_menu(screen, stars):
    # Прорисовка звезд
    for star in stars.sprites():
        star.draw()


def update_bullets(screen, bullets, comets, stats, score, coins, explosions):
    """Обновление движение пули"""
    W, H = screen.get_size()
    i = random.randint(0, 1)
    collision = pygame.mixer.Sound(
                "sounds/collision.mp3"
                )  # Звук столкновения пули и кометы

    bullets.update()
    explosions.update()

    for bullet in bullets.copy():
        if bullet.rect.bottom >= H:
            bullets.remove(bullet)

    for comet in comets:
        for bullet in bullets:
            # Проверка столкновения масок
            x = comet.rect.centerx
            y = comet.rect.centery

            offset = (comet.rect.x - bullet.rect.x,
                      comet.rect.y - bullet.rect.y)

            if bullet.mask.overlap_area(comet.mask, offset) > 9.2:
                collision.set_volume(stats.objectsvolume)
                collision.play()
                # Размер взрыва зависим от размера кометы
                if comet.size >= 0.6 and comet.size < 0.9:
                    expl = Explosion(screen, comet.rect.center, 'sm')
                if comet.size >= 0.9 and comet.size < 1.2:
                    expl = Explosion(screen, comet.rect.center, 'md')
                if comet.size >= 1.2 and comet.size < 1.5:
                    expl = Explosion(screen, comet.rect.center, 'lg')

                explosions.add(expl)

                if i == 1:
                    # Рандомное выпадение монет
                    new_coin = Coins(screen, x, y)
                    coins.add(new_coin)

                # Удаление объектов столкновения
                bullets.remove(bullet)
                comets.remove(comet)

        score.image_score()
        score.bullets_score()
        score.coins_score()


def bullets_movement(screen, player, bullets, stats, score):
    """Отслеживание полета пуль"""
    if stats.bullets_volume >= 1:
        new_bullet = Bullets(screen, player, stats)
        bullets.add(new_bullet)

        bul_add = pygame.mixer.Sound("sounds/bullet.mp3")  # Звук стрельбы
        bul_add.set_volume(stats.objectsvolume)
        bul_add.play()

        stats.bullets_volume -= 1


def update_comets(screen, comets, player, bullets, stats, score,
                  stars, addbullets, coins, explosions):
    """Обновление комет"""
    comets.update()
    stars.update()

    # Удаление объектов за пределамми экрана
    for comet in comets.copy():
        if (comet.rect.top >= comet.screen_rect.bottom or
            comet.rect.right >= comet.screen_rect.right + 110 or
            comet.rect.left <= comet.screen_rect.left - 150):
            comets.remove(comet)

    for star in stars.copy():
        if star.rect.top >= star.screen_rect.bottom:
            stars.remove(star)

    for coin in coins.copy():
        if coin.rect.top >= coin.screen_rect.bottom:
            coins.remove(coin)

    for comet in comets:
        # Проверка столкновения кометы с игроком
        offset = (comet.rect.x - player.rect.x, comet.rect.y - player.rect.y)

        if player.mask.overlap_area(comet.mask, offset) > 9.2:
            comets_size = comet.size

            # Взрыв при столкновении с игроком
            if comet.size >= 0.6 and comet.size < 0.9:
                expl = Explosion(screen, comet.rect.center, 'sm')
            if comet.size >= 0.9 and comet.size < 1.2:
                expl = Explosion(screen, comet.rect.center, 'md')
            if comet.size >= 1.2 and comet.size < 1.5:
                expl = Explosion(screen, comet.rect.center, 'lg')
            explosions.add(expl)
            comets.remove(comet)

            collision = pygame.mixer.Sound(
                "sounds/player.mp3"
            )  # Звук столкновения с игроком
            collision.set_volume(stats.objectsvolume)
            collision.play()

            player_kill(screen, player, comets, bullets,
                        stats, comets_size, score)


def create_comets(screen, group, W):
    """Создание характеристик для кометы"""
    x = random.randint(0, W)
    rand = random.randint(0, 2)

    images = ["comet.png", "comet1.png",
              "comet2.png", "comet5.png"]  # Изображения комет
    image = [pygame.image.load("images/comets/" + path).convert_alpha()
             for path in images]
    img = random.randint(0, len(images) - 1)

    speedy = random.uniform(1.1, 1.6)
    speedx = random.uniform(0.25, 0.55)

    size = random.uniform(0.6, 1.5)
    deg = random.randint(10, 180)

    return Comets(x, screen, group, rand, speedy,
                  speedx, deg, size, image[img])


def player_kill(screen, player, comets, bullets, stats, comets_size, score):
    # Обработка смерти игрока
    update_hard(player, comets_size, stats, score)

    if stats.player_left < 1:
        stats.run_game = "end"
        loading(screen, stats)


def update_hard(player, comets_size, stats, score):
    # Обработка вычитания жизней от размер кометы
    if comets_size >= 0.6 and comets_size < 0.85:
        stats.player_left -= 10
        score.hard -= 15
    if comets_size >= 0.85 and comets_size < 1.05:
        stats.player_left -= 20
        score.hard -= 30    
    if comets_size >= 1.05 and comets_size < 1.25:
        stats.player_left -= 30
        score.hard -= 45
    if comets_size >= 1.25 and comets_size <= 1.5:
        stats.player_left -= 40
        score.hard -= 60


def pause(screen, stats, button, player):
    """ Пауза """
    W, H = screen.get_size()
    stats.pause = True

    start_button = button.Button(285, 45, screen)
    exit_button = button.Button(312, 45, screen)

    def exit():
        # Выход в гланое меню
        stats.run_game = "pause"
        stats.pause = False
        stats.i += 1

    def runs():
        # Продолжение игры
        stats.pause = False

    # Остановка игрока
    player.mtop = False
    player.mbottom = False
    player.mright = False
    player.mleft = False

    # Цикл
    while stats.pause:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    stats.pause = False
                if event.key == pygame.K_ESCAPE:
                    stats.pause = False

        start_button.draw(W//2, (H//2) - 100, "Продолжить", 65, runs)
        exit_button.draw(W//2, H//2, "Главное меню", 65, exit)

        button.print_text(screen, "Пауза", W // 3 + 50, H // 2 - 50, 50)
        pygame.display.update()

    if stats.i >= 1:
        loading(screen, stats)  # Вызов функции загрузки меню


def _stars(screen, stars, W):
    # Создание рандомной координаты x для звезд
    x = random.randint(0, W)
    new_star = Stars(screen, x)
    stars.add(new_star)


def create_bullets(screen, addbullets, W):
    """Создание характеристик для доп.пуль"""
    x = random.randint(0, W)
    rand = random.randint(0, 2)

    speedy = random.uniform(0.6, 1.0)
    speedx = random.uniform(0.1, 0.4)

    new_addbullet = AddBullets(screen, x, speedx, speedy, rand)
    addbullets.add(new_addbullet)


def update_player(screen, player, addbullets, stats, coins, score):
    """Создание collision для игрока и дополнений"""
    addbullets.update()
    coins.update()

    bytk = [1, 2, 3, 4, 5]
    rand = random.sample(bytk, 1)

    for addbullet in addbullets:
        # Столкновение с доп.пулями
        offset = (addbullet.rect.x - player.rect.x,
                  addbullet.rect.y - player.rect.y)

        if player.mask.overlap_area(addbullet.mask, offset) > 10:
            stats.bullets_volume += random.randint(10, 30)

            bul_add = pygame.mixer.Sound(
                "sounds/add_bl.mp3"
            )  # Звук подбора дополнения
            bul_add.set_volume(stats.objectsvolume)
            bul_add.play()

            addbullets.remove(addbullet)

    for coin in coins:
        # Столкновение игрока с монетой
        offset = (coin.rect.x - player.rect.x, coin.rect.y - player.rect.y)

        if player.mask.overlap_area(coin.mask, offset) > 10:

            bytk_add = pygame.mixer.Sound(
                "sounds/bytk.mp3"
            )  # звук подбора монеты
            bytk_add.set_volume(stats.objectsvolume)
            bytk_add.play()

            coins.remove(coin)

            stats.coin_check += int(*rand)
            stats.coin += int(*rand)


def loading(screen, stats):
    """Загрузочный экран меню"""
    W, H = screen.get_size()

    # Обновление значения бутка
    with open("files/btc.txt", "r") as file:
        coinsread = int(file.read())
        with open("files/btc.txt", "w") as files:
            files.write(str(coinsread + stats.coin_check))
            stats.coin_check = 0

    with open("files/rd.txt", "r") as file:
        record = int(file.read())
        if stats.score > record:
            with open("files/rd.txt", "w") as files:
                files.write(str(stats.score))

    # Подсказки    
    hints = ["Здоровье пополняется каждые 1000 очков.",
             "Собирайте бутки и покупайте улучшения в магазине.",
             "Подбирайте магазин, чтобы пополнить патроны",
             "Скины можно выбрать в магазине.",
             "Выполняйте цели и получайте новые скины.",
             "Следите за патронами, не будет их - не будет вас.",
             "Кометы разные размерами, отнимают разное кол-во здоровья.",
             "О создании игры, можно посмотреть в вкладке 'Автор.'",
             "Уничтожайте кометы и получайте бутки.",
             "Чем больше вы живы, тем больше вы зарабатываете.",
             "Индикатор жизней: жизни < 60 - оранжевый,\
              жизни < 30 - красный. Следите!!!"]

    text = random.sample(hints, 1)
    x = 0

    screen.fill((0, 0, 0))

    # Цикл
    while x <= 150:
        time.sleep(0.0055)
        x += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        button.print_text(screen, "Загрузка...", W//1.1, H//1.05, 30)
        button.print_text(screen, "Сохранение данных.", W//W, H//1.05, 30)
        button.print_text(screen, str(*text), W//3, H//2, 30)

        pygame.display.update()
        pygame.display.flip()
