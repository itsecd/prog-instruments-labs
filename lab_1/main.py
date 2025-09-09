import os
import sys
import random
import tkinter
from random import randrange, randint

from PIL import Image, ImageDraw, ImageFont
import pygame
import keyboard

from tkinter import (
    Tk, Entry, StringVar, Button, Label,
    Radiobutton, Scale, HORIZONTAL
)

import turtle
from turtle import showturtle, Turtle


main_window = tkinter.Tk()

main_window["bg"] = "darkgray"
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
P.S. Прописан только 1 вариант\n""")

    '''---------------------------------------------------------------'''
    while direction_choice != 'Жаркая пустыня' or direction_choice != '1' or direction_choice != 'Дремучие леса' or direction_choice != '2' or direction_choice != 'Высокие горы' or direction_choice != '3':
        if direction_choice == 'Жаркая пустыня' or direction_choice == '1':

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

            while action_choice != '1' or action_choice != '2':
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

                # ---
                print()
                while side_of_the_world_direction != '1' or side_of_the_world_direction != '2' or side_of_the_world_direction != '3' or side_of_the_world_direction != '4':
                    if side_of_the_world_direction == '1' or side_of_the_world_direction == '2' or side_of_the_world_direction == '4':
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

                    while riddle_attempt != 'да' or riddle_attempt != 'нет':
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
                        if answer_input == '5' or answer_input == 'пять':
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
                                # 4
                            print("Сумма цифр этого числа:")
                            summa = mystery_number // 100 + mystery_number % 10 + mystery_number % 100 // 10
                            print(summa)
                            # 5
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

                                        while choice_coconut != '1' or choice_coconut != '2':

                                            if choice_coconut == '1':

                                                def i():
                                                    turtle.up()
                                                    turtle.goto(25, 0)
                                                    turtle.down()
                                                    for w in get_coconut_window.winfo_children():
                                                        if w.winfo_class() == 'Button': w.destroy()
                                                    ba1 = Button(get_coconut_window, text='↑', bg='yellow', fg='brown',
                                                                 font='verdana 10', command=y).pack()
                                                    ba2 = Button(get_coconut_window, text='↓', bg='yellow', fg='brown',
                                                                 font='verdana 10').pack()
                                                    ba3 = Button(get_coconut_window, text='←', bg='yellow', fg='brown',
                                                                 font='verdana 10').pack()
                                                    ba4 = Button(get_coconut_window, text='→', bg='yellow', fg='brown',
                                                                 font='verdana 10').pack()

                                                def y():
                                                    turtle.up()
                                                    turtle.goto(25, 80)
                                                    turtle.down()
                                                    for w in get_coconut_window.winfo_children():
                                                        if w.winfo_class() == 'Button': w.destroy()
                                                    ba1 = Button(get_coconut_window, text='↑', bg='yellow', fg='brown',
                                                                 font='verdana 10').pack()
                                                    ba2 = Button(get_coconut_window, text='↓', bg='yellow', fg='brown',
                                                                 font='verdana 10', command=i).pack()
                                                    ba3 = Button(get_coconut_window, text='←', bg='yellow', fg='brown',
                                                                 font='verdana 10', command=u).pack()
                                                    ba4 = Button(get_coconut_window, text='→', bg='yellow', fg='brown',
                                                                 font='verdana 10', command=op).pack()

                                                def u():
                                                    turtle.up()
                                                    turtle.goto(-60, 80)
                                                    turtle.down()
                                                    for w in get_coconut_window.winfo_children():
                                                        if w.winfo_class() == 'Button': w.destroy()
                                                    ba1 = Button(get_coconut_window, text='↑', bg='yellow', fg='brown',
                                                                 font='verdana 10').pack()
                                                    ba2 = Button(get_coconut_window, text='↓', bg='yellow', fg='brown',
                                                                 font='verdana 10').pack()
                                                    ba3 = Button(get_coconut_window, text='←', bg='yellow', fg='brown',
                                                                 font='verdana 10').pack()
                                                    ba4 = Button(get_coconut_window, text='→', bg='yellow', fg='brown',
                                                                 font='verdana 10', command=o).pack()

                                                def o():
                                                    turtle.up()
                                                    turtle.goto(25, 80)
                                                    turtle.down()
                                                    for w in get_coconut_window.winfo_children():
                                                        if w.winfo_class() == 'Button': w.destroy()
                                                    ba1 = Button(get_coconut_window, text='↑', bg='yellow', fg='brown',
                                                                 font='verdana 10').pack()
                                                    ba2 = Button(get_coconut_window, text='↓', bg='yellow', fg='brown',
                                                                 font='verdana 10', command=i).pack()
                                                    ba3 = Button(get_coconut_window, text='←', bg='yellow', fg='brown',
                                                                 font='verdana 10', command=u).pack()
                                                    ba4 = Button(get_coconut_window, text='→', bg='yellow', fg='brown',
                                                                 font='verdana 10', command=op).pack()

                                                def op():
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

                                                    def ky():
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
                                                        root101 = tkinter.Tk()
                                                        root101["bg"] = "yellow"
                                                        root101.geometry('300x50')
                                                        li1 = Label(root101, text='O ⚔', bg='yellow', fg='brown',
                                                                    font='verdana 10').pack()
                                                        root101.mainloop()
                                                        print('Вы вышли из кузницы.')
                                                        input("Нажмите Enter, чтобы продолжить")
                                                        sword_yes = True
                                                        coconut_yes = True
                                                        if sword_yes == True and coconut_yes == True:
                                                            print(
                                                                'Вы купили всё необходимое для дальнейших странствий.')
                                                            option_for_the_night = input('''Выйдя из деревни, вы вдруг вспомнили, что вам надо где-то переночевать.
Было 2 варинта - либо лечь под пальмой, либо попросить у кого-нибудь ночлега.
Вы призадумались и решили.
Выберите из предложенных вариантов и введите вариант ответа:
1)Лечь под пальмой
2)Попросить у кого-нибудь ночлега
''')

                                                            def noch():
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

                                                                def ky():
                                                                    print(
                                                                        '''Вы вошли в кузницу. Знакомого вам жителя здесь не оказалось''')
                                                                    print('Вы вышли из кузницы.')

                                                                def pr():
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
                                                                    lbl = Label(window, text='''Каракал - это пустынная кошка. Легко убивает антилоп. Делать это хищнику позволяют не только мощная хватка и ловкость, но и размеры:
В длину каракал достигает 85-ти сантиметров. Высота животного составляет полметра. Окрас зверя песочный, шерсть короткая и мягкая. На ушах есть кисти из длинной ости. Это делает каракала похожим на рысь.
Пустынная рысь одиночка, активна ночью. С наступлением темноты хищник охотится на среднего размера млекопитающих, птиц, пресмыкающихся. Название каракал, можно перевести как «черное ухо».''').pack()

                                                                    window.mainloop()
                                                                    print(
                                                                        'Поздравляем! Вы прошли 1-ую часть странствий!')
                                                                    print(
                                                                        'Чтобы продолжить игру, выберите игру "Странствия 2"')
                                                                    input("\nНажмите Enter, чтобы выйти")
                                                                    sys.exit()

                                                                while choice_in_town != '1' or choice_in_town != '2' or choice_in_town != '3':
                                                                    if choice_in_town == '1':
                                                                        pr()
                                                                        break
                                                                    elif choice_in_town == '2':
                                                                        ky()
                                                                        direction_choice_town = input(
                                                                            'Выберите, куда вы хотите направиться дальше(для этого напишите цифру, под которой обозначено здание.')
                                                                        while direction_choice_town != '1' or direction_choice_town != '3':
                                                                            if direction_choice_town == '1':
                                                                                pr()
                                                                                break
                                                                            elif direction_choice_town == '3':
                                                                                print('''Вы попытались открыть дверь
Дверь заперта''')
                                                                                print(
                                                                                    'Вы вспомнили, что вам нужно в продовольственный магазин')
                                                                                pr()
                                                                                break
                                                                            else:
                                                                                direction_choice_town = input(
                                                                                    'Выберите, куда вы хотите направиться дальше(для этого напишите цифру, под которой обозначено здание.')
                                                                            break
                                                                    elif choice_in_town == '3':
                                                                        print('''Вы попытались открыть дверь
Дверь заперта''')
                                                                        while direction_choice_town_dlc != '1' or direction_choice_town_dlc != '2':
                                                                            if direction_choice_town_dlc == '1':
                                                                                pr()
                                                                                break

                                                                            elif direction_choice_town_dlc == '2':
                                                                                ky()

                                                                                direction_choice_town = input(
                                                                                    'Выберите, куда вы хотите направиться дальше(для этого напишите цифру, под которой обозначено здание.')
                                                                                while direction_choice_town != '1' or direction_choice_town != '3':
                                                                                    if direction_choice_town == '1':
                                                                                        pr()
                                                                                        break
                                                                                    elif direction_choice_town == '3':
                                                                                        print('''Вы попытались открыть дверь
Дверь заперта''')
                                                                                        print(
                                                                                            'Вы вспомнили, что вам нужно в продовольственный магазин')
                                                                                        pr()
                                                                                        break
                                                                                    else:
                                                                                        direction_choice_town = input(
                                                                                            'Выберите, куда вы хотите направиться дальше(для этого напишите цифру, под которой обозначено здание.')
                                                                                        break
                                                                            else:
                                                                                direction_choice_town_dlc = input(
                                                                                    'Выберите, куда вы хотите направиться дальше(для этого напишите цифру, под которой обозначено здание.')

                                                                            break
                                                                    elif choice_in_town != '1' or choice_in_town != '2' or choice_in_town != '3':
                                                                        choice_in_town = input(
                                                                            'Выберите, куда вы хотите направиться (для этого напишите цифру, под которой обозначено здание.')

                                                                    break

                                                            while option_for_the_night != '1' or option_for_the_night != '2':

                                                                if option_for_the_night == '1':
                                                                    print(
                                                                        'Вы решили прилечь на ночь под раскидистой пальмой недалеко от деревни.')
                                                                    noch()
                                                                    break
                                                                elif option_for_the_night == '2':
                                                                    print('''Вы долго скитались, ища у кого попросить ночлега, но так и не нашли его.
В итоге, Вы решили прилечь на ночь под раскидистой пальмой недалеко от деревни.''')
                                                                    noch()
                                                                    break

                                                                else:
                                                                    print('Выберите возможный вариант ответа')
                                                                    option_for_the_night = input('''Выйдя из деревни, вы вдруг вспомнили, что вам надо где-то переночевать.
Было 2 варинта - либо лечь под пальмой, либо попросить у кого-нибудь ночлега.
Вы призадумались и решили.
Выберите из предложенных вариантов и введите вариант ответа:
1)Лечь под пальмой
2)Попросить у кого-нибудь ночлега''')

                                                    def pr():
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
                                                        root100 = tkinter.Tk()
                                                        root100["bg"] = "yellow"
                                                        root100.geometry('300x50')
                                                        li1 = Label(root100, text='ooooo oooo O', bg='yellow',
                                                                    fg='brown', font='verdana 10').pack()
                                                        root100.mainloop()
                                                        print('Вы вышли из продуктовой лавки.')
                                                        coconut_yes = True

                                                    def ky2():
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
                                                        root101 = tkinter.Tk()
                                                        root101["bg"] = "yellow"
                                                        root101.geometry('300x50')
                                                        li1 = Label(root101, text='o ⚔', bg='yellow', fg='brown',
                                                                    font='verdana 10').pack()
                                                        root101.mainloop()
                                                        print('Вы вышли из кузницы.')
                                                        sword_yes = True
                                                        coconut_yes = True

                                                    def pr2():
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
                                                        root100 = tkinter.Tk()
                                                        root100["bg"] = "yellow"
                                                        root100.geometry('300x50')
                                                        li1 = Label(root100, text='O ⚔', bg='yellow', fg='brown',
                                                                    font='verdana 10').pack()
                                                        print('Вы вышли из продуктовой лавки.')
                                                        root100.mainloop()

                                                        input("Нажмите Enter, чтобы продолжить")

                                                        sword_yes = True
                                                        coconut_yes = True

                                                        if sword_yes == True and coconut_yes == True:
                                                            print(
                                                                'Вы купили всё необходимое для дальнейших странствий.')
                                                            option_for_the_night = input('''Выйдя из деревни, вы вдруг вспомнили, что вам надо где-то переночевать.
Было 2 варинта - либо лечь под пальмой, либо попросить у кого-нибудь ночлега.
Вы призадумались и решили.
Выберите из предложенных вариантов и введите вариант ответа:
1)Лечь под пальмой
2)Попросить у кого-нибудь ночлега
''')

                                                            def noch():
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

                                                                def ky():
                                                                    print(
                                                                        '''Вы вошли в кузницу. Знакомого вам жителя здесь не оказалось''')
                                                                    print('Вы вышли из кузницы.')

                                                                def pr():
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
                                                                    lbl = Label(window, text='''Каракал - это пустынная кошка. Легко убивает антилоп. Делать это хищнику позволяют не только мощная хватка и ловкость, но и размеры:
В длину каракал достигает 85-ти сантиметров. Высота животного составляет полметра. Окрас зверя песочный, шерсть короткая и мягкая. На ушах есть кисти из длинной ости. Это делает каракала похожим на рысь.
Пустынная рысь одиночка, активна ночью. С наступлением темноты хищник охотится на среднего размера млекопитающих, птиц, пресмыкающихся. Название каракал, можно перевести как «черное ухо».''').pack()

                                                                    window.mainloop()

                                                                    print(
                                                                        'Поздравляем! Вы прошли 1-ую часть странствий!')
                                                                    print(
                                                                        'Чтобы продолжить игру, выберите игру "Странствия 2"')
                                                                    input("\nНажмите Enter, чтобы выйти")
                                                                    sys.exit()

                                                                while choice_in_town != '1' or choice_in_town != '2' or choice_in_town != '3':
                                                                    if choice_in_town == '1':
                                                                        pr()
                                                                        break
                                                                    elif choice_in_town == '2':
                                                                        ky()
                                                                        direction_choice_town = input(
                                                                            'Выберите, куда вы хотите направиться дальше(для этого напишите цифру, под которой обозначено здание.')
                                                                        while direction_choice_town != '1' or direction_choice_town != '3':
                                                                            if direction_choice_town == '1':
                                                                                pr()
                                                                                break
                                                                            elif direction_choice_town == '3':
                                                                                print('''Вы попытались открыть дверь
Дверь заперта''')
                                                                                print(
                                                                                    'Вы вспомнили, что вам нужно в продовольственный магазин')
                                                                                pr()
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
                                                                                pr()
                                                                                break

                                                                            elif direction_choice_town_dlc == '2':
                                                                                ky()

                                                                                direction_choice_town = input(
                                                                                    'Выберите, куда вы хотите направиться дальше(для этого напишите цифру, под которой обозначено здание.')
                                                                                while direction_choice_town != '1' or direction_choice_town != '3':
                                                                                    if direction_choice_town == '1':
                                                                                        pr()
                                                                                        break
                                                                                    elif direction_choice_town == '3':
                                                                                        print('''Вы попытались открыть дверь
Дверь заперта''')
                                                                                        print(
                                                                                            'Вы вспомнили, что вам нужно в продовольственный магазин')
                                                                                        pr()
                                                                                        break
                                                                                    else:
                                                                                        direction_choice_town = input(
                                                                                            'Выберите, куда вы хотите направиться дальше(для этого напишите цифру, под которой обозначено здание.')
                                                                                        break
                                                                            else:
                                                                                direction_choice_town_dlc = input(
                                                                                    'Выберите, куда вы хотите направиться дальше(для этого напишите цифру, под которой обозначено здание.')

                                                                            break
                                                                    elif choice_in_town != '1' or choice_in_town != '2' or choice_in_town != '3':
                                                                        choice_in_town = input(
                                                                            'Выберите, куда вы хотите направиться (для этого напишите цифру, под которой обозначено здание.')

                                                                    break

                                                            while option_for_the_night != '1' or option_for_the_night != '2':

                                                                if option_for_the_night == '1':
                                                                    print(
                                                                        'Вы решили прилечь на ночь под раскидистой пальмой недалеко от деревни.')
                                                                    noch()
                                                                    break
                                                                elif option_for_the_night == '2':
                                                                    print('''Вы долго скитались, ища у кого попросить ночлега, но так и не нашли его.
В итоге, Вы решили прилечь на ночь под раскидистой пальмой недалеко от деревни.''')
                                                                    noch()
                                                                    break

                                                                else:
                                                                    print('Выберите возможный вариант ответа')
                                                                    option_for_the_night = input('''Выйдя из деревни, вы вдруг вспомнили, что вам надо где-то переночевать.
Было 2 варинта - либо лечь под пальмой, либо попросить у кого-нибудь ночлега.
Вы призадумались и решили.
Выберите из предложенных вариантов и введите вариант ответа:
1)Лечь под пальмой
2)Попросить у кого-нибудь ночлега''')

                                                    while choice_in_town != '1' or choice_in_town != '2' or choice_in_town != '3':
                                                        if choice_in_town == '1':
                                                            pr()
                                                            direction_choice_in_town = input(
                                                                'Выберите, куда вы хотите направиться дальше(для этого напишите цифру, под которой обозначено здание.')
                                                            while direction_choice_in_town != '2' or direction_choice_in_town != '3':
                                                                if direction_choice_in_town == '2':
                                                                    ky()
                                                                    break
                                                                elif direction_choice_in_town == '3':
                                                                    print('''Вы попытались открыть дверь
        Дверь заперта''')
                                                                    print('Вы вспомнили, что вам нужно в кузницу')
                                                                    ky()
                                                                    break
                                                                else:
                                                                    direction_choice_in_town = input(
                                                                        'Выберите, куда вы хотите направиться дальше(для этого напишите цифру, под которой обозначено здание.')
                                                            break

                                                        elif choice_in_town == '2':
                                                            ky2()
                                                            direction_choice_town = input(
                                                                'Выберите, куда вы хотите направиться дальше(для этого напишите цифру, под которой обозначено здание.')
                                                            while direction_choice_town != '1' or direction_choice_town != '3':
                                                                if direction_choice_town == '1':
                                                                    pr2()
                                                                    break
                                                                elif direction_choice_town == '3':
                                                                    print('''Вы попытались открыть дверь
        Дверь заперта''')
                                                                    print(
                                                                        'Вы вспомнили, что вам нужно в продовольственный магазин')
                                                                    pr2()
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
                                                                    pr()
                                                                    direction_choice_in_town = input(
                                                                        'Выберите, куда вы хотите направиться дальше(для этого напишите цифру, под которой обозначено здание.')
                                                                    while direction_choice_in_town != '2' or direction_choice_in_town != '3':
                                                                        if direction_choice_in_town == '2':
                                                                            ky()
                                                                            break
                                                                        elif direction_choice_in_town == '3':
                                                                            print('''Вы попытались открыть дверь
        Дверь заперта''')
                                                                            print(
                                                                                'Вы вспомнили, что вам нужно в кузницу')
                                                                            ky()
                                                                            break
                                                                        else:
                                                                            direction_choice_in_town = input(
                                                                                'Выберите, куда вы хотите направиться дальше(для этого напишите цифру, под которой обозначено здание.')
                                                                        break
                                                                if direction_choice_town_dlc == '2':
                                                                    ky2()

                                                                    direction_choice_town = input(
                                                                        'Выберите, куда вы хотите направиться дальше(для этого напишите цифру, под которой обозначено здание.')
                                                                    while direction_choice_town != '1' or direction_choice_town != '3':
                                                                        if direction_choice_town == '1':
                                                                            pr2()
                                                                            break
                                                                        elif direction_choice_town == '3':
                                                                            print('''Вы попытались открыть дверь
                Дверь заперта''')
                                                                            print(
                                                                                'Вы вспомнили, что вам нужно в продовольственный магазин')
                                                                            pr2()
                                                                            break
                                                                    else:
                                                                        direction_choice_town = input(
                                                                            'Выберите, куда вы хотите направиться дальше(для этого напишите цифру, под которой обозначено здание.')
                                                                else:
                                                                    direction_choice_town_dlc = input(
                                                                        'Выберите, куда вы хотите направиться дальше(для этого напишите цифру, под которой обозначено здание.')

                                                                break
                                                            break

                                                        elif choice_in_town != '1' or choice_in_town != '2' or choice_in_town != '3':
                                                            choice_in_town = input(
                                                                'Выберите, куда вы хотите направиться (для этого напишите цифру, под которой обозначено здание).')

                                                        break

                                                get_coconut_window = tkinter.Tk()
                                                get_coconut_window["bg"] = "yellow"
                                                get_coconut_window.geometry('100x200')

                                                showturtle()

                                                def v():
                                                    turtle.fillcolor("green")
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
                                                turtle.fillcolor("brown")
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

                                                v()

                                                turtle.fillcolor("brown")
                                                turtle.right(90)

                                                turtle.up()
                                                turtle.right(90)
                                                turtle.right(90)
                                                turtle.fillcolor("yellow")

                                                turtle.up()
                                                turtle.goto(25, 0)
                                                turtle.down()

                                                ba = Button(get_coconut_window, text='Достать кокос', bg='yellow', fg='brown',
                                                            font='verdana 5', command=i).pack()

                                                get_coconut_window.mainloop()
                                                break
                                            if choice_coconut == '2':
                                                print('Вам нужна была вода, вы же поленились её достать.')
                                                print('Последствия ужасны.')
                                                break

                                            if choice_coconut != '1' or choice_coconut != '2':
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


def q():
     main_window.destroy()


def s():
    for w in main_window.winfo_children():
        if w.winfo_class() == 'Button': w.destroy()
    for w in main_window.winfo_children():
        if w.winfo_class() == 'Label': w.destroy()
    lab2 = Label(main_window, text='''Разработчик-программист:
    Ярославцев М.В.
    (Ярославцев Максим Владимирович)''', bg='darkgray', fg='brown', font='verdana 14').pack()
    b45 = Button(main_window, text='Выход', bg='darkgray', fg='brown', font='verdana 14', command=k).pack()


def ks():
    for w in main_window.winfo_children():
        if w.winfo_class() == 'Button': w.destroy()
    for w in main_window.winfo_children():
        if w.winfo_class() == 'Label': w.destroy()
    lab2 = Label(main_window, text='''Разработчик-программист:
    Ярославцев М.В.
    (Ярославцев Максим Владимирович)''', bg='red', fg='black', font='verdana 14').pack()
    b45 = Button(main_window, text='Выход', bg='red', fg='black', font='verdana 14', command=fon).pack()


def qs():
    for w in main_window.winfo_children():
        if w.winfo_class() == 'Button': w.destroy()
    for w in main_window.winfo_children():
        if w.winfo_class() == 'Label': w.destroy()
    lab2 = Label(main_window, text='''Разработчик-программист:
    Ярославцев М.В.
    (Ярославцев Максим Владимирович)''', bg='lightblue', fg='black', font='verdana 14').pack()
    b45 = Button(main_window, text='Выход', bg='lightblue', fg='black', font='verdana 14', command=fon).pack()


def ps():
    for w in main_window.winfo_children():
        if w.winfo_class() == 'Button': w.destroy()
    for w in main_window.winfo_children():
        if w.winfo_class() == 'Label': w.destroy()
    lab2 = Label(main_window, text='''Разработчик-программист:
    Ярославцев М.В.
    (Ярославцев Максим Владимирович)''', bg='green', fg='black', font='verdana 14').pack()
    b45 = Button(main_window, text='Выход', bg='green', fg='black', font='verdana 14', command=fon).pack()


def k():
    for w in main_window.winfo_children():
        if w.winfo_class() == 'Button': w.destroy()
    for w in main_window.winfo_children():
        if w.winfo_class() == 'Label': w.destroy()
    l1 = Label(main_window, text="Меню", bg='darkgray', fg='brown', font='verdana 20').pack()
    b1 = Button(main_window, text='Начать игру', bg='darkgray', fg='brown', font='verdana 14', command=game).pack()
    b2 = Button(main_window, text='Настройки', bg='darkgray', fg='brown', font='verdana 14', command=n).pack()
    b23 = Button(main_window, text='Создатели', bg='darkgray', fg='brown', font='verdana 14', command=s).pack()
    b3 = Button(main_window, text='Выход', bg='darkgray', fg='brown', font='verdana 14', command=q).pack()


def fon():
    direction_choice = v1.get()
    if direction_choice == 0:
        main_window["bg"] = "red"
        for w in main_window.winfo_children():
            if w.winfo_class() == 'Radiobutton': w.destroy()
        for w in main_window.winfo_children():
            if w.winfo_class() == 'Label': w.destroy()
        for w in main_window.winfo_children():
            if w.winfo_class() == 'Button': w.destroy()
        l1 = Label(main_window, text="Меню", bg='red', fg='black', font='verdana 20').pack()
        b1 = Button(main_window, text='Начать игру', bg='darkgray', fg='brown', font='verdana 14', command=game).pack()
        b2 = Button(main_window, text='Настройки', bg='darkgray', fg='brown', font='verdana 14', command=n).pack()
        b23 = Button(main_window, text='Создатели', bg='darkgray', fg='brown', font='verdana 14', command=ks).pack()
        b3 = Button(main_window, text='Выход', bg='darkgray', fg='brown', font='verdana 14', command=q).pack()
    elif direction_choice == 1:
        main_window["bg"] = "lightblue"
        for w in main_window.winfo_children():
            if w.winfo_class() == 'Radiobutton': w.destroy()
        for w in main_window.winfo_children():
            if w.winfo_class() == 'Label': w.destroy()
        for w in main_window.winfo_children():
            if w.winfo_class() == 'Button': w.destroy()
        l1 = Label(main_window, text="Меню", bg='lightblue', fg='brown', font='verdana 20').pack()
        b1 = Button(main_window, text='Начать игру', bg='darkgray', fg='brown', font='verdana 14', command=game).pack()
        b2 = Button(main_window, text='Настройки', bg='darkgray', fg='brown', font='verdana 14', command=n).pack()
        b23 = Button(main_window, text='Создатели', bg='darkgray', fg='brown', font='verdana 14', command=qs).pack()
        b3 = Button(main_window, text='Выход', bg='darkgray', fg='brown', font='verdana 14', command=q).pack()
    elif direction_choice == 2:
        main_window["bg"] = "green"
        for w in main_window.winfo_children():
            if w.winfo_class() == 'Radiobutton': w.destroy()
        for w in main_window.winfo_children():
            if w.winfo_class() == 'Label': w.destroy()
        for w in main_window.winfo_children():
            if w.winfo_class() == 'Button': w.destroy()
        l1 = Label(main_window, text="Меню", bg='green', fg='brown', font='verdana 20').pack()
        b1 = Button(main_window, text='Начать игру', bg='darkgray', fg='brown', font='verdana 14', command=game).pack()
        b2 = Button(main_window, text='Настройки', bg='darkgray', fg='brown', font='verdana 14', command=n).pack()
        b23 = Button(main_window, text='Создатели', bg='darkgray', fg='brown', font='verdana 14', command=ps).pack()
        b3 = Button(main_window, text='Выход', bg='darkgray', fg='brown', font='verdana 14', command=q).pack()
    else:
        main_window["bg"] = "darkgray"
        for w in main_window.winfo_children():
            if w.winfo_class() == 'Radiobutton': w.destroy()
        for w in main_window.winfo_children():
            if w.winfo_class() == 'Label': w.destroy()
        for w in main_window.winfo_children():
            if w.winfo_class() == 'Button': w.destroy()
        l1 = Label(main_window, text="Меню", bg='darkgray', fg='brown', font='verdana 20').pack()
        b1 = Button(main_window, text='Начать игру', bg='darkgray', fg='brown', font='verdana 14', command=game).pack()
        b2 = Button(main_window, text='Настройки', bg='darkgray', fg='brown', font='verdana 14', command=n).pack()
        b23 = Button(main_window, text='Создатели', bg='darkgray', fg='brown', font='verdana 14', command=s).pack()
        b3 = Button(main_window, text='Выход', bg='darkgray', fg='brown', font='verdana 14', command=q).pack()


def n():
    for w in main_window.winfo_children():
        if w.winfo_class() == 'Button': w.destroy()
    for w in main_window.winfo_children():
        if w.winfo_class() == 'Label': w.destroy()
    lab2 = Label(main_window, text="Цвет фона", font='verdana 14').pack()
    v1.set(4)
    r1 = Radiobutton(main_window, text='Красный', fg='red', variable=v1, value=0).pack()
    r2 = Radiobutton(main_window, text='Светлоголубой', fg='lightblue', variable=v1, value=1).pack()
    r3 = Radiobutton(main_window, text='Зелёный', fg='green', variable=v1, value=2).pack()
    r4 = Radiobutton(main_window, text='Серый', fg='darkgray', variable=v1, value=3).pack()
    but2 = Button(main_window, text='Поменять', command=fon).pack()


# виджеты
l1 = Label(main_window, text="Меню", bg='darkgray', fg='brown', font='verdana 20').pack()
print("Добро пожаловать в игру 'Приключения'!");
b1 = Button(main_window, text='Начать игру', bg='darkgray', fg='brown', font='verdana 14', command=game).pack()
b2 = Button(main_window, text='Настройки', bg='darkgray', fg='brown', font='verdana 14', command=n).pack()
b23 = Button(main_window, text='Создатели', bg='darkgray', fg='brown', font='verdana 14', command=s).pack()
b3 = Button(main_window, text='Выход', bg='darkgray', fg='brown', font='verdana 14', command=q).pack()

# ---

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
