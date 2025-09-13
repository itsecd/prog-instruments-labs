import os
import random
import sys
import tkinter
from random import randint, randrange
from tkinter import (
    Button, Entry, HORIZONTAL, Label,
    Radiobutton, Scale, StringVar, Tk
)

import keyboard
import pygame
import turtle
from PIL import Image, ImageDraw, ImageFont
from turtle import Turtle, showturtle

from constants import *


main_window = tkinter.Tk()

main_window["bg"] = BG_DARKGRAY
main_window.geometry('500x300')

text = StringVar()
v1 = tkinter.IntVar()
v2 = tkinter.IntVar()


def game():
    main_window.destroy()
    name = input('Здравствуйте, как мне вас называть? ')
    print('Ясненько, вас, значит, кличут', name)
    print('Ну что ж,', name, ',приступим-с.')
    direction_choice = input("""Куда вы хотите направиться?
Можете выбрать из предложенных:
1)Жаркая пустыня
2)Дремучие леса
3)Высокие горы
Пишите понятно, я привык к хорошей речи (можете написать номер варианта)
P.S. Прописан только первый вариант\n""")

    '''---------------------------------------------------------------'''
    while True:
        if direction_choice in ['Жаркая пустыня', '1']:

            print("""Вы очутились в жаркой и душной пустыне.
Кругом - ни души.
Надо что-то делать.""")
            print()
            img_of_desert = Image.open('Images for Python/pust.jpg')
            img_of_desert.show()
            input('Нажмите Enter, чтобы продолжить')
            action_choice = input('''Что же вы предпримете?
Можете выбрать из предложенных:
1)Идти куда глаза глядят, доверясь провидению.
2)Сдаться, решив, что вы не в силах выпутаться из этой ситуации.
(Напишите вариант выбранного ответа)''')
            print()

            while True:
                if action_choice == '1':
                    pygame.init()
                    pygame.mixer.music.load('music/put_in_arab_gorod.mp3')
                    pygame.mixer.music.play(-1)
                    print('''Вы решили продолжить путешествие, скитаясь по жаркой и душной пустыне.
Ну что ж,''', name, ''',раз уж вы не сдаётесь, у вас есть шанс выбраться из пустыни, иными словами - у вас есть шанс на жизнь.
''')
                    break

                elif action_choice == '2':
                    print('Вы сдались, ваше путешествие окончено.')
                    break

                else:
                    print('Выберите возможный вариант ответа')
                    action_choice = input('''Что же вы предпримите?
Можете выбрать из предложенных:
1)Идти куда глаза глядят, доверясь провидению.
2)Сдаться, решив, что вы не в силах выпутаться из этой ситуации.
(Напишите вариант выбранного ответа)''')

            if action_choice == '1':

                print('Итак,', name, ''',вы выбрали путь истинный.
Сдаётся лишь человек отчаившийся. А вы, я смотрю, не из таких, ''', name, '.'
                                                                          '''Вы долгое время скитались по пустыне в поисках поселения или, хотя бы, воды.
                                                                          Вдруг вдалеке вы увидели оазис. Обрадовавшись, вы побежали, подскальзываясь на песке, но оазис не приближался.
                                                                          Он оставался на месте. Пробежав километр-другой (по крайней мере, вам так казалось), вы поняли, что это лишь галлюцинации из-за недостатка воды.
                                                                          Нужно было искать воду. Идти можно было на все четыре стороны. Какую вы выберете?''')
                side_of_the_world_direction = input('''Можете выбрать из предложенных:
1)Север
2)Юг
3)Запад
4)Восток
(Напишите вариант выбранного ответа)''')

                print()
                while True:
                    if side_of_the_world_direction in ['1', '2', '4']:
                        print('''К вашему глубочайшему сожалению, вы не смогли найти источник воды.
Ваше же путешествие окончилось для вас, господин''', name, ',весьма печально.')
                        break
                    elif side_of_the_world_direction == '3':
                        print('Ура!')
                        break
                    else:
                        print('Выберите возможный вариант ответа')
                        side_of_the_world_direction = input('''Можете выбрать из предложенных:
1)Север
2)Юг
3)Запад 
4)Восток
(Напишите вариант выбранного ответа)''')
                print()
                if side_of_the_world_direction == '3':
                    print('''Вы совершенно случайно наткнулись на оазис - "островок жизни" в пустыне !
Вам необычайно повезло! Увидя оазис, вы сообразили, что вода должна быть где-то рядом.
Вдруг к вам навстречу из-за высокой пальмы вышел человек. Не прошло и пары минут, как вы оказались лицом к лицу.
Этот человек был невысокого роста, темнокожий с голубыми глазами.
Житель оазиса произнёс:
"Добро пожаловать в наше уединённое место. Вы должны отгадать загадку, прежде чем войти в наш 'кусочек жизни'."
"Как ваше имя?" - спросил он и вы ответили ему: "Моё имя -''', name, '"')
                    print('''"Что ж,''', name, ''',ещё раз здравствуйте." - сказал житель оазиса пустыни.
"Чтобы пройти в наш оазис, вы должны отгадать три загадки"''')
                    riddle_attempt = input('Попытаетесь ли вы отгадать их? Ответьте: "да" или "нет".')
                    riddle_attempt = riddle_attempt.lower()

                    while True:
                        if riddle_attempt == 'да':
                            break
                        elif riddle_attempt  == 'нет':
                            print('Ну что ж,', name, ',до свидания')
                            break
                        else:
                            riddle_attempt = input('Попытаетесь ли вы отгадать загадки? Ответьте чётко: "да" или "нет".')
                            riddle_attempt = riddle_attempt.lower()
                    if riddle_attempt == 'да':
                        print('''Первая загадка представляет собой стихотворение. Оно гласит:
Я считаю дырки в сыре,
Три плюс два равно ...''')
                        img_of_cheese = Image.open('Images for Python/sur.jpg')
                        img_of_cheese.show()
                        answer_input = input('Скажите ваш ответ.')
                        if answer_input in ['5', 'пять']:
                            print('Ваш ответ оказался верным. Необычный вы человек,', name, '.')
                            print('Что ж, приступим ко второй загадке,', name, '.')
                            print('''Во второй загадке вы должны отгадать число.''')
                            mystery_number = random.randrange(1000)
                            print("Я загадал число. Сейчас я кое-что про него Вам расскажу")
                            print("В этом числе:")

                            if 0 <= mystery_number <= 9:
                                print("1 знак")
                            elif 10 <= mystery_number <= 99:
                                print("2 знакa")
                            else:
                                print("3 знакa")

                            print("Оно заканчивается цифрой", mystery_number % 10)

                            if mystery_number % 2 == 0:
                                print("Это число чётное")
                            else:
                                print("Это число нечётное")

                            print("Сумма цифр этого числа:")
                            summa = mystery_number // 100 + mystery_number % 10 + mystery_number % 100 // 10
                            print(summa)

                            units_digit = mystery_number % 10
                            tens_digit = mystery_number % 100 // 10
                            hundreds_digit = mystery_number // 10

                            if 0 <= mystery_number <= 9:
                                print("Цифра только одна")
                            elif 10 <= mystery_number <= 99:
                                if units_digit == tens_digit:
                                    print("Все цифры этого числа одинаковые")
                                else:
                                    print("Все цифры этого числа неодинаковые")
                            else:
                                if units_digit == tens_digit and tens_digit == hundreds_digit:
                                    print("Все цифры этого числа одинаковые")
                                else:
                                    print("Цифры этого числа неодинаковые")

                            print("Угадаешь с 10 раз, какое число я загадал?")

                            for i in range(1, 11):
                                user = int(input('Введите число '))
                                if mystery_number == user:
                                    print("Угадал")
                                    print('Вы отгадали? Это редкость. Что ж, приступим к третьей загадке.')
                                    print('Я загадал число от 1 до 20. Попробуй его отгадать за 6 попыток.')
                                    number = random.randint(1, 20)
                                    for i in range(1, 7):
                                        print('Попытка №', i)
                                        igrok = int(input('Ваше предложение?'))
                                        print(igrok)
                                        if number > igrok:
                                            print('Моё число больше')
                                        elif number < igrok:
                                            print('Моё число меньше')
                                        else:
                                            break

                                    if i <= 6 and number == igrok:
                                        print('Поздравляю!')
                                        print('Я загадал число:', number)
                                        print('Количество сделанных попыток:', i)
                                    else:
                                        print('К сожалению,ты не справился...')
                                        print('Было загадано число', number)
                                        break

                                    if number == igrok:
                                        print('Дорогой', name, ', вы профессионал.')
                                        print('''- Прошу проходить в наш уголок рая - пригласил вас пройти внутрь оазиса его житель.
Пройдя пару-тройку местров, вы, открыв рот, уставились на пальмы, с вершин которых поигрывали налитостью кокосы.
- Можно мне попробовать один? - скромно спросили вы.
- А почему бы нет? Только его ещё достать надо.
- Как залезать-то?
- Обезьяны по деревьям прыгают и ты сможешь.
 Вы подумали-подумали и решили, что раз пить хочется, надо попытаться достать заветный кокос.''')
                                        choice_coconut = input('''Выберете из предложенных вариантов:
1)Доставать кокос
2)Не доставать кокос
''')

                                        while True:

                                            if choice_coconut == '1':

                                                def down():
                                                    turtle.up()
                                                    turtle.goto(25, 0)
                                                    turtle.down()
                                                    for w in get_coconut_window.winfo_children():
                                                        if w.winfo_class() == 'Button': w.destroy()
                                                    Button(get_coconut_window, text='↑', bg=BG_YELLOW, fg=FG_BROWN,
                                                                 font='verdana 10', command=up).pack()
                                                    Button(get_coconut_window, text='↓', bg=BG_YELLOW, fg=FG_BROWN,
                                                                 font='verdana 10').pack()
                                                    Button(get_coconut_window, text='←', bg=BG_YELLOW, fg=FG_BROWN,
                                                                 font='verdana 10').pack()
                                                    Button(get_coconut_window, text='→', bg=BG_YELLOW, fg=FG_BROWN,
                                                                 font='verdana 10').pack()

                                                def up():
                                                    turtle.up()
                                                    turtle.goto(25, 80)
                                                    turtle.down()
                                                    for w in get_coconut_window.winfo_children():
                                                        if w.winfo_class() == 'Button': w.destroy()
                                                    Button(get_coconut_window, text='↑', bg=BG_YELLOW, fg=FG_BROWN,
                                                                 font='verdana 10').pack()
                                                    Button(get_coconut_window, text='↓', bg=BG_YELLOW, fg=FG_BROWN,
                                                                 font='verdana 10', command=down).pack()
                                                    Button(get_coconut_window, text='←', bg=BG_YELLOW, fg=FG_BROWN,
                                                                 font='verdana 10', command=left).pack()
                                                    Button(get_coconut_window, text='→', bg=BG_YELLOW, fg=FG_BROWN,
                                                                 font='verdana 10', command=get_coconut).pack()

                                                def left():
                                                    turtle.up()
                                                    turtle.goto(-60, 80)
                                                    turtle.down()
                                                    for w in get_coconut_window.winfo_children():
                                                        if w.winfo_class() == 'Button': w.destroy()
                                                    Button(get_coconut_window, text='↑', bg=BG_YELLOW, fg=FG_BROWN,
                                                                 font='verdana 10').pack()
                                                    Button(get_coconut_window, text='↓', bg=BG_YELLOW, fg=FG_BROWN,
                                                                 font='verdana 10').pack()
                                                    Button(get_coconut_window, text='←', bg=BG_YELLOW, fg=FG_BROWN,
                                                                 font='verdana 10').pack()
                                                    Button(get_coconut_window, text='→', bg=BG_YELLOW, fg=FG_BROWN,
                                                                 font='verdana 10', command=right).pack()

                                                def right():
                                                    turtle.up()
                                                    turtle.goto(25, 80)
                                                    turtle.down()
                                                    for w in get_coconut_window.winfo_children():
                                                        if w.winfo_class() == 'Button': w.destroy()
                                                    Button(get_coconut_window, text='↑', bg=BG_YELLOW, fg=FG_BROWN,
                                                                 font='verdana 10').pack()
                                                    Button(get_coconut_window, text='↓', bg=BG_YELLOW, fg=FG_BROWN,
                                                                 font='verdana 10', command=down).pack()
                                                    Button(get_coconut_window, text='←', bg=BG_YELLOW, fg=FG_BROWN,
                                                                 font='verdana 10', command=left).pack()
                                                    Button(get_coconut_window, text='→', bg=BG_YELLOW, fg=FG_BROWN,
                                                                 font='verdana 10', command=get_coconut).pack()

                                                def get_coconut():
                                                    turtle.up()
                                                    turtle.goto(100, 80)
                                                    turtle.down()
                                                    turtle.begin_fill()
                                                    turtle.fillcolor("lemonchiffon")
                                                    turtle.circle(15)
                                                    turtle.end_fill()
                                                    print('Круто!')
                                                    print('Кокос в ваших руках!')
                                                    turtle.up()
                                                    turtle.goto(25, 0)
                                                    turtle.down()
                                                    showturtle()
                                                    get_coconut_window.destroy()
                                                    turtle.bye()
                                                    img_of_coconut = Image.open('Images for Python/cocos.jpg')
                                                    img_of_coconut.show()
                                                    input('Нажмите Enter, чтобы продолжить')
                                                    print(
                                                        'Вы смогли достать кокос - это возвысило вас в глазах жителей оазиса.')
                                                    print('''К вам обратился житель оазиса, встретивший вас:
"Я смотрю Вы человек находчивый.
Ну что ж, тогда располагайтесь у нас в деревне, вот вам 10 монет на покупку всего необходимого" - Сказал житель и протянул руку с деньгами и запиской.
Вы взяли их и прошли за ним в деревню.''')
                                                    Inventory = ('10 монет')
                                                    print('Вы располагаете следующими вещами: ', Inventory)
                                                    print('''Записка гласила:
Я думаю, вам необходимо приобрести пропитание и оружие для дальнейших странствий.''')
                                                    img_of_map_of_oasis = Image.open('Images for Python/map_for_oasis.jpg')
                                                    img_of_map_of_oasis.show()
                                                    choice_in_town = input(
                                                        'Выберите, куда вы хотите направиться (для этого напишите цифру, под которой обозначено здание).')

                                                    def forge():
                                                        print('Вы вошли в кузницу')
                                                        img_of_forge = Image.open('Images for Python/kyz.jpg')
                                                        img_of_forge.show()
                                                        input('Нажмите Enter, чтобы продолжить')
                                                        print('''Вы можете приобрести: меч
В наличии: 10 шт.''')
                                                        print(
                                                            'Заглянув в "список", который Вам дал житель оазиса, вы вспомнили, что вам надо купить пропитание и оружие.')
                                                        input('''"Какое количество мечей Вы хотите приобрести?" - оборвал ваши раздумья продавец магазина.
Вспомнив, что у вас осталось лишь 9 монет, Вы сказали: "Мне нужно купить один меч."
(Для продолжения нажмите "Enter") ''')

                                                        print(
                                                            '"С Вас девять монет" - сказал продавец и Вы отдали ему горстку монет, получив меч.')
                                                        print('Вы приобрели 1 меч')
                                                        img_of_sword = Image.open('Images for Python/sword.jpg')
                                                        img_of_sword.show()
                                                        input('Нажмите Enter, чтобы продолжить')
                                                        Inventory = ('1 кокос, 1 меч')
                                                        print('Вы располагаете следующими вещами: ', Inventory)
                                                        window_inventory = tkinter.Tk()
                                                        window_inventory["bg"] = BG_YELLOW
                                                        window_inventory.geometry('300x50')
                                                        Label(window_inventory, text='O ⚔', bg=BG_YELLOW, fg=FG_BROWN,
                                                                    font='verdana 10').pack()
                                                        window_inventory.mainloop()
                                                        print('Вы вышли из кузницы.')
                                                        input("Нажмите Enter, чтобы продолжить")
                                                        sword_yes = True
                                                        coconut_yes = True

                                                        if sword_yes and coconut_yes:
                                                            print(
                                                                'Вы купили всё необходимое для дальнейших странствий.')
                                                            option_for_the_night = input('''Выйдя из деревни, вы вдруг вспомнили, что вам надо где-то переночевать.
Было 2 варинта - либо лечь под пальмой, либо попросить у кого-нибудь ночлега.
Вы призадумались и решили.
Выберите из предложенных вариантов и введите вариант ответа:
1)Лечь под пальмой
2)Попросить у кого-нибудь ночлега
''')

                                                            def night():
                                                                print('''Всё было как нельзя лучше - всё-таки под открытым небом и на свежем воздухе,
как вдруг вы, засыпая, услышали шорох в соседних кустах. Вы пристально посмотрели в то место и через некоторое время из кустов выпрыгнул зверь, очень похожий на рысь.
Откуда рыси в здешних краях - вы даже успели подумать над этим вопросом.''')
                                                                input('Для продолжения нажмите "Enter"')
                                                                img_of_caracal = Image.open('Images for Python/caracal.jpg')
                                                                img_of_caracal.show()
                                                                input('Для продолжения нажмите "Enter"')
                                                                print(
                                                                    '''Но потом, осмыслив ситуацию и осознав, что думать над происхождением животного - неуместно, вы быстро встали и побежали по напралению к деревне.''')

                                                                print('''Животное, похожее на рысь, посмотрело вам вслед и убежало дальше.
"Странно" - подумали вы, - "Вроде напала, а не погналась вслед."
Прибежав в деревню, вы начали искать уже знакомого вам жителя этого оазиса.''')
                                                                print('Вы вошли в деревню.')
                                                                img_of_map_of_oasis = Image.open(
                                                                    'Images for Python/map_for_oasis.jpg')
                                                                img_of_map_of_oasis.show()
                                                                choice_in_town = input(
                                                                    'Выберите, куда вы хотите направиться (для этого напишите цифру, под которой обозначено здание.')

                                                                def forge():
                                                                    print(
                                                                        '''Вы вошли в кузницу. Знакомого вам жителя здесь не оказалось''')
                                                                    print('Вы вышли из кузницы.')

                                                                def grocery():
                                                                    print('''Вы вошли в продуктовую лавку.''')
                                                                    print(
                                                                        'Спустя 5 минут безудержных поисков, вы наткнулись на него в продовольственном магазине.')
                                                                    print(
                                                                        'Вы вышли из продуктовой лавки, а затем и из деревни.')
                                                                    print(
                                                                        '''Вы вышли из деревни, за вами вышел знакомый вам житель оазиса. ''')
                                                                    print('''"На меня вроде напала рысь, но эта была не рысь, и она за мной не погналась" - вы быстро протараторили вашему знакомому.
- Это и не рысь. Это каракал - пустынная кошка. Она хоть и хищница, но людьми не питается, - просмеялся ваш знакомый, - так что нечего было её бояться.
- Фуух... Спасибо. А как вас зовут?
- Акиль. Вас, кажется,''', name, '?')
                                                                    print('Да - улыбнулись вы.')
                                                                    print('Вы изучили объект под названием "каракал"')
                                                                    input('Для продолжения нажмите "Enter"')
                                                                    img_caracal = Image.open(
                                                                        'Images for Python/caracal_2.jpg')
                                                                    img_caracal.show()
                                                                    window = Tk()
                                                                    window.title("Каракал")
                                                                    Label(window, text='''Каракал - это пустынная кошка. Легко убивает антилоп. Делать это хищнику позволяют не только мощная хватка и ловкость, но и размеры:
В длину каракал достигает 85-ти сантиметров. Высота животного составляет полметра. Окрас зверя песочный, шерсть короткая и мягкая. На ушах есть кисти из длинной ости. Это делает каракала похожим на рысь.
Пустынная рысь одиночка, активна ночью. С наступлением темноты хищник охотится на среднего размера млекопитающих, птиц, пресмыкающихся. Название каракал, можно перевести как «черное ухо».''').pack()

                                                                    window.mainloop()
                                                                    print(
                                                                        'Поздравляем! Вы прошли 1-ую часть странствий!')
                                                                    print(
                                                                        'Чтобы продолжить игру, выберите игру "Странствия 2"')
                                                                    input("\nНажмите Enter, чтобы выйти")
                                                                    sys.exit()

                                                                while True:
                                                                    if choice_in_town == '1':
                                                                        grocery()
                                                                        break
                                                                    elif choice_in_town == '2':
                                                                        forge()
                                                                        direction_choice_town = input(
                                                                            'Выберите, куда вы хотите направиться дальше(для этого напишите цифру, под которой обозначено здание.')
                                                                        while True:
                                                                            if direction_choice_town == '1':
                                                                                grocery()
                                                                                break
                                                                            elif direction_choice_town == '3':
                                                                                print('''Вы попытались открыть дверь
Дверь заперта''')
                                                                                print(
                                                                                    'Вы вспомнили, что вам нужно в продовольственный магазин')
                                                                                grocery()
                                                                                break
                                                                            else:
                                                                                direction_choice_town = input(
                                                                                    'Выберите, куда вы хотите направиться дальше(для этого напишите цифру, под которой обозначено здание.')
                                                                        break
                                                                    elif choice_in_town == '3':
                                                                        print('''Вы попытались открыть дверь
Дверь заперта''')
                                                                        direction_choice_town_dlc = input(
                                                                            'Выберите, куда вы хотите направиться дальше(для этого напишите цифру, под которой обозначено здание.')
                                                                        while True:
                                                                            if direction_choice_town_dlc == '1':
                                                                                grocery()
                                                                                break

                                                                            elif direction_choice_town_dlc == '2':
                                                                                forge()

                                                                                direction_choice_town = input(
                                                                                    'Выберите, куда вы хотите направиться дальше(для этого напишите цифру, под которой обозначено здание.')
                                                                                while True:
                                                                                    if direction_choice_town == '1':
                                                                                        grocery()
                                                                                        break
                                                                                    elif direction_choice_town == '3':
                                                                                        print('''Вы попытались открыть дверь
Дверь заперта''')
                                                                                        print(
                                                                                            'Вы вспомнили, что вам нужно в продовольственный магазин')
                                                                                        grocery()
                                                                                        break
                                                                                    else:
                                                                                        direction_choice_town = input(
                                                                                            'Выберите, куда вы хотите направиться дальше(для этого напишите цифру, под которой обозначено здание.')
                                                                                break
                                                                            else:
                                                                                direction_choice_town_dlc = input(
                                                                                    'Выберите, куда вы хотите направиться дальше(для этого напишите цифру, под которой обозначено здание.')

                                                                        break
                                                                    else:
                                                                        choice_in_town = input(
                                                                            'Выберите, куда вы хотите направиться (для этого напишите цифру, под которой обозначено здание.')

                                                            while True:

                                                                if option_for_the_night == '1':
                                                                    print(
                                                                        'Вы решили прилечь на ночь под раскидистой пальмой недалеко от деревни.')
                                                                    night()
                                                                    break
                                                                elif option_for_the_night == '2':
                                                                    print('''Вы долго скитались, ища у кого попросить ночлега, но так и не нашли его.
В итоге, Вы решили прилечь на ночь под раскидистой пальмой недалеко от деревни.''')
                                                                    night()
                                                                    break

                                                                else:
                                                                    print('Выберите возможный вариант ответа')
                                                                    option_for_the_night = input('''Выйдя из деревни, вы вдруг вспомнили, что вам надо где-то переночевать.
Было 2 варинта - либо лечь под пальмой, либо попросить у кого-нибудь ночлега.
Вы призадумались и решили.
Выберите из предложенных вариантов и введите вариант ответа:
1)Лечь под пальмой
2)Попросить у кого-нибудь ночлега''')

                                                    def grocery():
                                                        print('''Вы вошли в продуктовую лавку''')
                                                        img_of_grocery = Image.open('Images for Python/prod.jpg')
                                                        img_of_grocery.show()
                                                        input('Нажмите Enter, чтобы продолжить')
                                                        print('''Вы можете приобрести: кокосы
В наличии: 1 шт.''')
                                                        print(
                                                            'Заглянув в "список", который Вам дал житель оазиса, вы вспомнили, что вам надо купить пропитание и оружие.')
                                                        input('''"Какое количество кокосов Вы хотите приобрести?" - оборвал ваши раздумья продавец магазина.
Вспомнив, что в данном магазине есть всего лишь 1 кокос, Вы сказали: "Мне нужно купить один кокос."
(Для продолжения нажмите "Enter") ''')

                                                        print(
                                                            '"С Вас одна монета" - сказал продавец и Вы отдали ему монету, получив кокос.')
                                                        print('Вы приобрели 1 кокос')
                                                        img_of_coconut = Image.open('Images for Python/cocos_in_magaz.jpg')
                                                        img_of_coconut.show()
                                                        input('Нажмите Enter, чтобы продолжить')
                                                        Inventory = ('9 монет, 1 кокос')
                                                        print('Вы располагаете следующими вещами: ', Inventory)
                                                        window_inventory = tkinter.Tk()
                                                        window_inventory["bg"] = BG_YELLOW
                                                        window_inventory.geometry('300x50')
                                                        Label(window_inventory, text='ooooo oooo O', bg=BG_YELLOW,
                                                                    fg=FG_BROWN, font='verdana 10').pack()
                                                        window_inventory.mainloop()
                                                        print('Вы вышли из продуктовой лавки.')
                                                        coconut_yes = True

                                                    def forge_result():
                                                        print('''Вы вошли в кузницу''')
                                                        img_of_forge = Image.open('Images for Python/kyz.jpg')
                                                        img_of_forge.show()
                                                        input('Нажмите Enter, чтобы продолжить')
                                                        print('''Вы можете приобрести: меч.
В наличии: 10 шт.''')
                                                        print(
                                                            'Заглянув в "список", который Вам дал житель оазиса, вы вспомнили, что вам надо купить пропитание и оружие.')
                                                        input('''"Какое количество мечей Вы хотите приобрести?" - оборвал ваши раздумья продавец магазина.
Вспомнив, что у вас осталось лишь 9 монет, Вы сказали: "Мне нужно купить один меч."
(Для продолжения нажмите "Enter") ''')

                                                        print(
                                                            '"С Вас девять монет" - сказал продавец и Вы отдали ему горстку монет, получив меч.')
                                                        print('Вы приобрели 1 меч')
                                                        img_of_sword = Image.open('Images for Python/sword.jpg')
                                                        img_of_sword.show()
                                                        input('Нажмите Enter, чтобы продолжить')
                                                        Inventory = ('1 монета, 1 меч')
                                                        print('Вы располагаете следующими вещами: ', Inventory)
                                                        window_inventory = tkinter.Tk()
                                                        window_inventory["bg"] = BG_YELLOW
                                                        window_inventory.geometry('300x50')
                                                        Label(window_inventory, text='o ⚔', bg=BG_YELLOW, fg=FG_BROWN,
                                                                    font='verdana 10').pack()
                                                        window_inventory.mainloop()
                                                        print('Вы вышли из кузницы.')
                                                        sword_yes = True
                                                        coconut_yes = True

                                                    def grocery_result():
                                                        print('''Вы вошли в продуктовую лавку''')
                                                        img_of_grocery = Image.open('Images for Python/prod.jpg')
                                                        img_of_grocery.show()
                                                        input('Нажмите Enter, чтобы продолжить')
                                                        print('''Вы можете приобрести: кокосы
В наличии: 1 шт.''')
                                                        print(
                                                            'Заглянув в "список", который Вам дал житель оазиса, вы вспомнили, что вам надо купить пропитание и оружие.')
                                                        input('''"Какое количество кокосов Вы хотите приобрести?" - оборвал ваши раздумья продавец магазина.
Вспомнив, что в данном магазине есть всего лишь 1 кокос, Вы сказали: "Мне нужно купить один кокос."
(Для продолжения нажмите "Enter") ''')

                                                        print(
                                                            '"С Вас одна монета" - сказал продавец и Вы отдали ему монету, получив кокос.')
                                                        print('Вы приобрели 1 кокос')
                                                        img_of_coconut = Image.open('Images for Python/cocos_in_magaz.jpg')
                                                        img_of_coconut.show()
                                                        input('Нажмите Enter, чтобы продолжить')
                                                        Inventory = ('1 кокос, 1 меч')
                                                        print('Вы располагаете следующими вещами: ', Inventory)
                                                        window_inventory = tkinter.Tk()
                                                        window_inventory["bg"] = BG_YELLOW
                                                        window_inventory.geometry('300x50')
                                                        Label(window_inventory, text='O ⚔', bg=BG_YELLOW, fg=FG_BROWN,
                                                                    font='verdana 10').pack()
                                                        print('Вы вышли из продуктовой лавки.')
                                                        window_inventory.mainloop()

                                                        input("Нажмите Enter, чтобы продолжить")

                                                        sword_yes = True
                                                        coconut_yes = True

                                                        if sword_yes and coconut_yes:
                                                            print(
                                                                'Вы купили всё необходимое для дальнейших странствий.')
                                                            option_for_the_night = input('''Выйдя из деревни, вы вдруг вспомнили, что вам надо где-то переночевать.
Было 2 варинта - либо лечь под пальмой, либо попросить у кого-нибудь ночлега.
Вы призадумались и решили.
Выберите из предложенных вариантов и введите вариант ответа:
1)Лечь под пальмой
2)Попросить у кого-нибудь ночлега
''')

                                                            def night():
                                                                print('''Всё было как нельзя лучше - всё-таки под открытым небом и на свежем воздухе,
как вдруг вы, засыпая, услышали шорох в соседних кустах. Вы пристально посмотрели в то место и через некоторое время из кустов выпрыгнул зверь, очень похожий на рысь.
Откуда рыси в здешних краях - вы даже успели подумать над этим вопросом.''')
                                                                input('Для продолжения нажмите "Enter"')
                                                                img_of_caracal = Image.open('Images for Python/caracal.jpg')
                                                                img_of_caracal.show()
                                                                input('Для продолжения нажмите "Enter"')
                                                                print(
                                                                    '''Но потом, осмыслив ситуацию и осознав, что думать над происхождением животного - неуместно, вы быстро встали и побежали по напралению к деревне.''')
                                                                print('''Животное, похожее на рысь, посмотрело вам вслед и убежало дальше.
"Странно" - подумали вы, - "Вроде напала, а не погналась вслед."
Прибежав в деревню, вы начали искать уже знакомого вам жителя этого оазиса.''')
                                                                print('Вы вошли в деревню.')
                                                                img_of_map_of_oasis = Image.open('Images for Python/map_for_oasis.jpg')
                                                                img_of_map_of_oasis.show()
                                                                choice_in_town = input(
                                                                    'Выберите, куда вы хотите направиться (для этого напишите цифру, под которой обозначено здание.')

                                                                def forge():
                                                                    print(
                                                                        '''Вы вошли в кузницу. Знакомого вам жителя здесь не оказалось''')
                                                                    print('Вы вышли из кузницы.')

                                                                def grocery():
                                                                    print('''Вы вошли в продуктовую лавку.''')
                                                                    print(
                                                                        'Спустя 5 минут безудержных поисков, вы наткнулись на него в продовольственном магазине.')
                                                                    print(
                                                                        'Вы вышли из продуктовой лавки, а затем и из деревни.')
                                                                    print(
                                                                        '''Вы вышли из деревни, за вами вышел знакомый вам житель оазиса. ''')
                                                                    print('''"На меня вроде напала рысь, но эта была не рысь, и она за мной не погналась" - вы быстро протараторили вашему знакомому.
- Это и не рысь. Это каракал - пустынная кошка. Она хоть и хищница, но людьми не питается, - просмеялся ваш знакомый, - так что нечего было её бояться.
- Фуух... Спасибо. А как вас зовут?
- Акиль. Вас, кажется,''', name, '?')
                                                                    print('Да - улыбнулись вы.')
                                                                    print('Вы изучили объект под названием "каракал"')
                                                                    input('Для продолжения нажмите "Enter"')
                                                                    img_caracal = Image.open(
                                                                        'Images for Python/caracal_2.jpg')
                                                                    img_caracal.show()

                                                                    window = Tk()
                                                                    window.title("Каракал")
                                                                    Label(window, text='''Каракал - это пустынная кошка. Легко убивает антилоп. Делать это хищнику позволяют не только мощная хватка и ловкость, но и размеры:
В длину каракал достигает 85-ти сантиметров. Высота животного составляет полметра. Окрас зверя песочный, шерсть короткая и мягкая. На ушах есть кисти из длинной ости. Это делает каракала похожим на рысь.
Пустынная рысь одиночка, активна ночью. С наступлением темноты хищник охотится на среднего размера млекопитающих, птиц, пресмыкающихся. Название каракал, можно перевести как «черное ухо».''').pack()

                                                                    window.mainloop()

                                                                    print(
                                                                        'Поздравляем! Вы прошли 1-ую часть странствий!')
                                                                    print(
                                                                        'Чтобы продолжить игру, выберите игру "Странствия 2"')
                                                                    input("\nНажмите Enter, чтобы выйти")
                                                                    sys.exit()

                                                                while True:
                                                                    if choice_in_town == '1':
                                                                        grocery()
                                                                        break
                                                                    elif choice_in_town == '2':
                                                                        forge()
                                                                        direction_choice_town = input(
                                                                            'Выберите, куда вы хотите направиться дальше(для этого напишите цифру, под которой обозначено здание.')
                                                                        while True:
                                                                            if direction_choice_town == '1':
                                                                                grocery()
                                                                                break
                                                                            elif direction_choice_town == '3':
                                                                                print('''Вы попытались открыть дверь
Дверь заперта''')
                                                                                print(
                                                                                    'Вы вспомнили, что вам нужно в продовольственный магазин')
                                                                                grocery()
                                                                                break
                                                                            else:
                                                                                direction_choice_town = input(
                                                                                    'Выберите, куда вы хотите направиться дальше(для этого напишите цифру, под которой обозначено здание.')
                                                                        break
                                                                    elif choice_in_town == '3':
                                                                        print('''Вы попытались открыть дверь
Дверь заперта''')
                                                                        direction_choice_town_dlc = input(
                                                                            'Выберите, куда вы хотите направиться дальше(для этого напишите цифру, под которой обозначено здание.')
                                                                        while True:
                                                                            if direction_choice_town_dlc == '1':
                                                                                grocery()
                                                                                break

                                                                            elif direction_choice_town_dlc == '2':
                                                                                forge()

                                                                                direction_choice_town = input(
                                                                                    'Выберите, куда вы хотите направиться дальше(для этого напишите цифру, под которой обозначено здание.')
                                                                                while True:
                                                                                    if direction_choice_town == '1':
                                                                                        grocery()
                                                                                        break
                                                                                    elif direction_choice_town == '3':
                                                                                        print('''Вы попытались открыть дверь
Дверь заперта''')
                                                                                        print(
                                                                                            'Вы вспомнили, что вам нужно в продовольственный магазин')
                                                                                        grocery()
                                                                                        break
                                                                                    else:
                                                                                        direction_choice_town = input(
                                                                                            'Выберите, куда вы хотите направиться дальше(для этого напишите цифру, под которой обозначено здание.')
                                                                                break
                                                                            else:
                                                                                direction_choice_town_dlc = input(
                                                                                    'Выберите, куда вы хотите направиться дальше(для этого напишите цифру, под которой обозначено здание.')

                                                                        break
                                                                    else:
                                                                        choice_in_town = input(
                                                                            'Выберите, куда вы хотите направиться (для этого напишите цифру, под которой обозначено здание.')

                                                            while True:

                                                                if option_for_the_night == '1':
                                                                    print(
                                                                        'Вы решили прилечь на ночь под раскидистой пальмой недалеко от деревни.')
                                                                    night()
                                                                    break
                                                                elif option_for_the_night == '2':
                                                                    print('''Вы долго скитались, ища у кого попросить ночлега, но так и не нашли его.
В итоге, Вы решили прилечь на ночь под раскидистой пальмой недалеко от деревни.''')
                                                                    night()
                                                                    break

                                                                else:
                                                                    print('Выберите возможный вариант ответа')
                                                                    option_for_the_night = input('''Выйдя из деревни, вы вдруг вспомнили, что вам надо где-то переночевать.
Было 2 варинта - либо лечь под пальмой, либо попросить у кого-нибудь ночлега.
Вы призадумались и решили.
Выберите из предложенных вариантов и введите вариант ответа:
1)Лечь под пальмой
2)Попросить у кого-нибудь ночлега''')

                                                    while True:
                                                        if choice_in_town == '1':
                                                            grocery()
                                                            direction_choice_in_town = input(
                                                                'Выберите, куда вы хотите направиться дальше(для этого напишите цифру, под которой обозначено здание.')
                                                            while True:
                                                                if direction_choice_in_town == '2':
                                                                    forge()
                                                                    break
                                                                elif direction_choice_in_town == '3':
                                                                    print('''Вы попытались открыть дверь
        Дверь заперта''')
                                                                    print('Вы вспомнили, что вам нужно в кузницу')
                                                                    forge()
                                                                    break
                                                                else:
                                                                    direction_choice_in_town = input(
                                                                        'Выберите, куда вы хотите направиться дальше(для этого напишите цифру, под которой обозначено здание.')
                                                            break

                                                        elif choice_in_town == '2':
                                                            forge_result()
                                                            direction_choice_town = input(
                                                                'Выберите, куда вы хотите направиться дальше(для этого напишите цифру, под которой обозначено здание.')
                                                            while True:
                                                                if direction_choice_town == '1':
                                                                    grocery_result()
                                                                    break
                                                                elif direction_choice_town == '3':
                                                                    print('''Вы попытались открыть дверь
        Дверь заперта''')
                                                                    print(
                                                                        'Вы вспомнили, что вам нужно в продовольственный магазин')
                                                                    grocery_result()
                                                                    break
                                                                else:
                                                                    direction_choice_town = input(
                                                                        'Выберите, куда вы хотите направиться дальше(для этого напишите цифру, под которой обозначено здание.')
                                                            break

                                                        elif choice_in_town == '3':
                                                            print('''Вы попытались открыть дверь
        Дверь заперта''')
                                                            direction_choice_town_dlc = input(
                                                                'Выберите, куда вы хотите направиться дальше(для этого напишите цифру, под которой обозначено здание.')
                                                            while direction_choice_town_dlc != '1' or direction_choice_town_dlc != '2':
                                                                if direction_choice_town_dlc == '1':
                                                                    grocery()
                                                                    direction_choice_in_town = input(
                                                                        'Выберите, куда вы хотите направиться дальше(для этого напишите цифру, под которой обозначено здание.')
                                                                    while direction_choice_in_town != '2' or direction_choice_in_town != '3':
                                                                        if direction_choice_in_town == '2':
                                                                            forge()
                                                                            break
                                                                        elif direction_choice_in_town == '3':
                                                                            print('''Вы попытались открыть дверь
        Дверь заперта''')
                                                                            print(
                                                                                'Вы вспомнили, что вам нужно в кузницу')
                                                                            forge()
                                                                            break
                                                                        else:
                                                                            direction_choice_in_town = input(
                                                                                'Выберите, куда вы хотите направиться дальше(для этого напишите цифру, под которой обозначено здание.')
                                                                    break
                                                                if direction_choice_town_dlc == '2':
                                                                    forge_result()

                                                                    direction_choice_town = input(
                                                                        'Выберите, куда вы хотите направиться дальше(для этого напишите цифру, под которой обозначено здание.')
                                                                    while direction_choice_town != '1' or direction_choice_town != '3':
                                                                        if direction_choice_town == '1':
                                                                            grocery_result()
                                                                            break
                                                                        elif direction_choice_town == '3':
                                                                            print('''Вы попытались открыть дверь
                Дверь заперта''')
                                                                            print(
                                                                                'Вы вспомнили, что вам нужно в продовольственный магазин')
                                                                            grocery_result()
                                                                            break
                                                                        else:
                                                                            direction_choice_town = input(
                                                                                'Выберите, куда вы хотите направиться дальше(для этого напишите цифру, под которой обозначено здание.')
                                                                    break
                                                                else:
                                                                    direction_choice_town_dlc = input(
                                                                        'Выберите, куда вы хотите направиться дальше(для этого напишите цифру, под которой обозначено здание.')
                                                            break

                                                        else:
                                                            choice_in_town = input(
                                                                'Выберите, куда вы хотите направиться (для этого напишите цифру, под которой обозначено здание).')


                                                get_coconut_window = tkinter.Tk()
                                                get_coconut_window["bg"] = BG_YELLOW
                                                get_coconut_window.geometry('100x200')

                                                showturtle()

                                                def tree():
                                                    turtle.fillcolor(TURTLE_GREEN)
                                                    turtle.begin_fill()
                                                    turtle.fd(80)
                                                    turtle.right(90)
                                                    turtle.fd(130)
                                                    turtle.right(90)
                                                    turtle.fd(210)
                                                    turtle.right(90)
                                                    turtle.fd(130)
                                                    turtle.right(90)
                                                    turtle.end_fill()

                                                turtle.up()
                                                turtle.goto(0, -100)
                                                turtle.down()
                                                turtle.begin_fill()
                                                turtle.fillcolor(TURTLE_BROWN)
                                                turtle.left(90)
                                                turtle.fd(230)
                                                turtle.right(90)
                                                turtle.fd(50)
                                                turtle.right(90)
                                                turtle.fd(230)
                                                turtle.right(90)
                                                turtle.fd(50)
                                                turtle.end_fill()

                                                turtle.up()
                                                turtle.right(90)
                                                turtle.fd(230)
                                                turtle.right(90)
                                                turtle.fd(50)
                                                turtle.down()

                                                tree()

                                                turtle.fillcolor(TURTLE_BROWN)
                                                turtle.right(90)

                                                turtle.up()
                                                turtle.right(90)
                                                turtle.right(90)
                                                turtle.fillcolor(TURTLE_YELLOW)

                                                turtle.up()
                                                turtle.goto(25, 0)
                                                turtle.down()

                                                Button(get_coconut_window, text='Достать кокос', bg=BG_YELLOW, fg=FG_BROWN,
                                                            font='verdana 5', command=down).pack()

                                                get_coconut_window.mainloop()
                                                break
                                            if choice_coconut == '2':
                                                print('Вам нужна была вода, вы же поленились её достать.')
                                                print('Последствия ужасны.')
                                                break

                                            if choice_coconut in ['1', '2']:
                                                choice_coconut = input('''Выберете из предложенных вариантов:
1)Доставать кокос
2)Не доставать кокос
''')

                                        break

                                else:
                                    print("Не угадал")

                        else:
                            break
                break

            if action_choice == '2':
                break

        elif direction_choice == 'Дремучие леса' or direction_choice == '2':

            print('''Вы очутились в дремучих, непроглядных лесах.
Осмотрясь, вы пришли к выводу, что в радиусе дюжины метров никого и ничего нет.
(К вашему глубочайшему сожалению, лес был настóлько непроходимым, что вы видили лишь на двенадцать метров)
Задумавшись над происходящим, вы подумали: "Что же делать?".
Что вы решаете предпринять?''')
            print()
            break
        # ---

        elif direction_choice == 'Высокие горы' or direction_choice == '3':

            print('''Вы очутились в высоких-высоких горах.
Посмотревши вокруг, вы обнаружили, что лишь орлы летают над вами,
а людей не видно было далеко-далеко, насколько можно было окинуть взором эту возвышенность, на которой вы оказались.
Посидевши на камне, вы поняли: для того, чтобы выбраться из гор, вам надо было что-то предпринять.
Что же вы решаете предпринять?''')
            print()
            break

        else:
            print('Выберите возможный вариант ответа')
            direction_choice = input("""Куда вы хотите направиться?
Можете выбрать из предложенных:
1)Жаркая пустыня
2)Дремучие леса
3)Высокие горы
Пишите понятно, я привык к хорошей речи (можете написать номер варианта)""")
            print()

    input("\nНажмите Enter, чтобы выйти")


def exit():
     main_window.destroy()


def developer_darkgray_brown():
    for w in main_window.winfo_children():
        if w.winfo_class() == 'Button': w.destroy()
    for w in main_window.winfo_children():
        if w.winfo_class() == 'Label': w.destroy()
    Label(main_window, text='''Разработчик-программист:
    Ярославцев М.В.
    (Ярославцев Максим Владимирович)''', bg=BG_DARKGRAY, fg=FG_BROWN, font='verdana 14').pack()
    Button(main_window, text='Выход', bg=BG_DARKGRAY, fg=FG_BROWN, font='verdana 14', command=menu).pack()


def developer_red_black():
    for w in main_window.winfo_children():
        if w.winfo_class() == 'Button': w.destroy()
    for w in main_window.winfo_children():
        if w.winfo_class() == 'Label': w.destroy()
    Label(main_window, text='''Разработчик-программист:
    Ярославцев М.В.
    (Ярославцев Максим Владимирович)''', bg=BG_RED, fg=FG_BLACK, font='verdana 14').pack()
    Button(main_window, text='Выход', bg=BG_RED, fg=FG_BLACK, font='verdana 14', command=main_color).pack()


def developer_lightblue_black():
    for w in main_window.winfo_children():
        if w.winfo_class() == 'Button': w.destroy()
    for w in main_window.winfo_children():
        if w.winfo_class() == 'Label': w.destroy()
    Label(main_window, text='''Разработчик-программист:
    Ярославцев М.В.
    (Ярославцев Максим Владимирович)''', bg=BG_LIGHTBLUE, fg=FG_BLACK, font='verdana 14').pack()
    Button(main_window, text='Выход', bg=BG_LIGHTBLUE, fg=FG_BLACK, font='verdana 14', command=main_color).pack()


def developer_green_black():
    for w in main_window.winfo_children():
        if w.winfo_class() == 'Button': w.destroy()
    for w in main_window.winfo_children():
        if w.winfo_class() == 'Label': w.destroy()
    Label(main_window, text='''Разработчик-программист:
    Ярославцев М.В.
    (Ярославцев Максим Владимирович)''', bg=BG_GREEN, fg=FG_BLACK, font='verdana 14').pack()
    Button(main_window, text='Выход', bg=BG_GREEN, fg=FG_BLACK, font='verdana 14', command=main_color).pack()


def menu():
    for w in main_window.winfo_children():
        if w.winfo_class() == 'Button': w.destroy()
    for w in main_window.winfo_children():
        if w.winfo_class() == 'Label': w.destroy()
    Label(main_window, text="Меню", bg=BG_DARKGRAY, fg=FG_BROWN, font='verdana 20').pack()
    Button(main_window, text='Начать игру', bg=BG_DARKGRAY, fg=FG_BROWN, font='verdana 14', command=game).pack()
    Button(main_window, text='Настройки', bg=BG_DARKGRAY, fg=FG_BROWN, font='verdana 14', command=color_background).pack()
    Button(main_window, text='Создатели', bg=BG_DARKGRAY, fg=FG_BROWN, font='verdana 14', command=developer_darkgray_brown).pack()
    Button(main_window, text='Выход', bg=BG_DARKGRAY, fg=FG_BROWN, font='verdana 14', command=exit).pack()


def main_color():
    direction_choice = v1.get()
    if direction_choice == 0:
        main_window["bg"] = BG_RED
        for w in main_window.winfo_children():
            if w.winfo_class() == 'Radiobutton': w.destroy()
        for w in main_window.winfo_children():
            if w.winfo_class() == 'Label': w.destroy()
        for w in main_window.winfo_children():
            if w.winfo_class() == 'Button': w.destroy()
        Label(main_window, text="Меню", bg=BG_RED, fg=FG_BLACK, font='verdana 20').pack()
        Button(main_window, text='Начать игру', bg=BG_DARKGRAY, fg=FG_BROWN, font='verdana 14', command=game).pack()
        Button(main_window, text='Настройки', bg=BG_DARKGRAY, fg=FG_BROWN, font='verdana 14', command=color_background).pack()
        Button(main_window, text='Создатели', bg=BG_DARKGRAY, fg=FG_BROWN, font='verdana 14', command=developer_red_black).pack()
        Button(main_window, text='Выход', bg=BG_DARKGRAY, fg=FG_BROWN, font='verdana 14', command=exit).pack()
    elif direction_choice == 1:
        main_window["bg"] = BG_LIGHTBLUE
        for w in main_window.winfo_children():
            if w.winfo_class() == 'Radiobutton': w.destroy()
        for w in main_window.winfo_children():
            if w.winfo_class() == 'Label': w.destroy()
        for w in main_window.winfo_children():
            if w.winfo_class() == 'Button': w.destroy()
        Label(main_window, text="Меню", bg=BG_LIGHTBLUE, fg=FG_BROWN, font='verdana 20').pack()
        Button(main_window, text='Начать игру', bg=BG_DARKGRAY, fg=FG_BROWN, font='verdana 14', command=game).pack()
        Button(main_window, text='Настройки', bg=BG_DARKGRAY, fg=FG_BROWN, font='verdana 14', command=color_background).pack()
        Button(main_window, text='Создатели', bg=BG_DARKGRAY, fg=FG_BROWN, font='verdana 14', command=developer_lightblue_black).pack()
        Button(main_window, text='Выход', bg=BG_DARKGRAY, fg=FG_BROWN, font='verdana 14', command=exit).pack()
    elif direction_choice == 2:
        main_window["bg"] = BG_GREEN
        for w in main_window.winfo_children():
            if w.winfo_class() == 'Radiobutton': w.destroy()
        for w in main_window.winfo_children():
            if w.winfo_class() == 'Label': w.destroy()
        for w in main_window.winfo_children():
            if w.winfo_class() == 'Button': w.destroy()
        Label(main_window, text="Меню", bg=BG_GREEN, fg=FG_BROWN, font='verdana 20').pack()
        Button(main_window, text='Начать игру', bg=BG_DARKGRAY, fg=FG_BROWN, font='verdana 14', command=game).pack()
        Button(main_window, text='Настройки', bg=BG_DARKGRAY, fg=FG_BROWN, font='verdana 14', command=color_background).pack()
        Button(main_window, text='Создатели', bg=BG_DARKGRAY, fg=FG_BROWN, font='verdana 14', command=developer_green_black).pack()
        Button(main_window, text='Выход', bg=BG_DARKGRAY, fg=FG_BROWN, font='verdana 14', command=exit).pack()
    else:
        main_window["bg"] = BG_DARKGRAY
        for w in main_window.winfo_children():
            if w.winfo_class() == 'Radiobutton': w.destroy()
        for w in main_window.winfo_children():
            if w.winfo_class() == 'Label': w.destroy()
        for w in main_window.winfo_children():
            if w.winfo_class() == 'Button': w.destroy()
        Label(main_window, text="Меню", bg=BG_DARKGRAY, fg=FG_BROWN, font='verdana 20').pack()
        Button(main_window, text='Начать игру', bg=BG_DARKGRAY, fg=FG_BROWN, font='verdana 14', command=game).pack()
        Button(main_window, text='Настройки', bg=BG_DARKGRAY, fg=FG_BROWN, font='verdana 14', command=color_background).pack()
        Button(main_window, text='Создатели', bg=BG_DARKGRAY, fg=FG_BROWN, font='verdana 14', command=developer_darkgray_brown).pack()
        Button(main_window, text='Выход', bg=BG_DARKGRAY, fg=FG_BROWN, font='verdana 14', command=exit).pack()


def color_background():
    for w in main_window.winfo_children():
        if w.winfo_class() == 'Button': w.destroy()
    for w in main_window.winfo_children():
        if w.winfo_class() == 'Label': w.destroy()
    Label(main_window, text="Цвет фона", font='verdana 14').pack()
    v1.set(4)
    Radiobutton(main_window, text='Красный', fg=FG_RED, variable=v1, value=0).pack()
    Radiobutton(main_window, text='Светлоголубой', fg=FG_LIGHTBLUE, variable=v1, value=1).pack()
    Radiobutton(main_window, text='Зелёный', fg=FG_GREEN, variable=v1, value=2).pack()
    Radiobutton(main_window, text='Серый', fg=FG_DARKGRAY, variable=v1, value=3).pack()
    Button(main_window, text='Поменять', command=main_color).pack()


Label(main_window, text="Меню", bg=BG_DARKGRAY, fg=FG_BROWN, font='verdana 20').pack()
print("Добро пожаловать в игру 'Приключения'!")
Button(main_window, text='Начать игру', bg=BG_DARKGRAY, fg=FG_BROWN, font='verdana 14', command=game).pack()
Button(main_window, text='Настройки', bg=BG_DARKGRAY, fg=FG_BROWN, font='verdana 14', command=color_background).pack()
Button(main_window, text='Создатели', bg=BG_DARKGRAY, fg=FG_BROWN, font='verdana 14', command=developer_darkgray_brown).pack()
Button(main_window, text='Выход', bg=BG_DARKGRAY, fg=FG_BROWN, font='verdana 14', command=exit).pack()

scale_window = tkinter.Tk()


def getV(event):
    an = scale1.get()
    print("Ваш возраст: ", an)

    if an > 5:
        scale_window.destroy()

    if an > 5:
        print('Доступ к игре открыт')

    if an <= 5:
        scale_window.destroy()
        main_window.destroy()

    if an <= 5:
        print('Доступ к игре закрыт')


scale1 = Scale(scale_window, orient=HORIZONTAL, length=1000, from_=0, to=100, tickinterval=5,
               resolution=5)
button1 = Button(scale_window, text=u"Подтвердите ваш возраст")
scale1.pack()
button1.pack()
button1.bind("<Button-1>", getV)

scale_window.mainloop()
main_window.mainloop()
