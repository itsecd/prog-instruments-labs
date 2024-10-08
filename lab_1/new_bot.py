import csv
import os.path
import time

import telebot
import config
import get_schedule as gs
import get_schedule_session as gs_session
from telebot import types

bot = telebot.TeleBot(config.TOKEN)


@bot.message_handler(commands=['start'])
def welcome(message):
    """
    Handles the /start command.
    Sends a welcome message and information about the bot.
    """
    list_info = []
    with open('data/info.txt', encoding='utf-8') as file:
        list_info = file.readlines()

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    button_change = types.KeyboardButton('/change')
    markup.add(button_change)

    sti = open('data/stickers/HI.webp', 'rb')
    bot.send_sticker(message.chat.id, sti)

    bot.send_message(
        message.chat.id, 'Приветик, {0.first_name}!\nЯ бот Эдди, '
                         'призванный помогать в учёбе\n'.format(message.from_user))

    bot.send_message(
        message.chat.id, list_info)

    received_message_text = bot.send_message(
        message.chat.id, 'Нажми "/change" чтобы начать', reply_markup=markup)

    bot.register_next_step_handler(received_message_text, change_option)


@bot.message_handler(commands=['change'])
def change_option(message):
    """
    Handles the /change command.
    Allows the user to change settings or access features.
    """
    write_in_file = True
    with open('users.csv', 'r', encoding='utf-8') as file:
        reader_der = csv.reader(file, delimiter=';')
        for row in reader_der:
            if row[1] == str(message.from_user.id):
                write_in_file = False
    if write_in_file:
        print(
            f'Новый пользователь -> {message.from_user.first_name} '
            f'-> ID: {message.from_user.id}')
        with open('users.csv', 'a', newline='', encoding='utf-8') as file:
            printer = csv.writer(file, delimiter=';')
            printer.writerow([
                message.from_user.first_name,
                message.from_user.id,
                message.chat.id
            ])

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    button_lab_assignment = types.KeyboardButton('Узнать задание по лабе')
    button_get_book = types.KeyboardButton('Достать учебник')
    button_get_secret = types.KeyboardButton('Получить секретик')
    button_get_schedule = types.KeyboardButton('Узнать расписание')
    button_useful_links = types.KeyboardButton('Важные ссылки')
    if str(message.from_user.id) == '765103434':
        button_news = types.KeyboardButton('News')
        markup.add(button_lab_assignment, button_get_book, button_get_secret,
                   button_get_schedule, button_useful_links, button_news)
    else:
        markup.add(button_lab_assignment, button_get_book, button_get_secret,
                   button_get_schedule, button_useful_links)

    received_message_text = bot.send_message(
        message.chat.id, 'Погнали!', reply_markup=markup)
    bot.register_next_step_handler(received_message_text, expanded_change)


@bot.message_handler(content_types=['text'])
def expanded_change(message):
    """
    Handles text messages from users.
    Executes corresponding actions based on the message text.
    """
    if message.chat.type == 'private':
        if message.text == 'Узнать задание по лабе':
            sticker_really = open('data/stickers/REALLY.webp', 'rb')
            bot.send_sticker(message.chat.id, sticker_really)
            markup = types.ReplyKeyboardRemove()
            list_items = []
            list_files = os.listdir('data/labs_book/labs/')
            list_files.sort()
            for files in list_files:
                list_items.append(files)
            markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
            for item in list_items:
                markup.add(item)
            item = types.KeyboardButton('Вернуться в меню')
            markup.add(item)
            received_message_text = bot.send_message(
                message.chat.id, 'Что именно нужно?', reply_markup=markup)
            bot.register_next_step_handler(received_message_text, change_lab_task)

        elif message.text == 'Достать учебник':
            sticker_yes = open('data/stickers/YES.webp', 'rb')
            bot.send_sticker(message.chat.id, sticker_yes)
            markup = types.ReplyKeyboardRemove()
            bot.send_message(message.chat.id, 'Окей', reply_markup=markup)

            list_items = []
            list_files = os.listdir('data/labs_book/')
            list_files.sort()
            for files in list_files:
                list_items.append(files)
            markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
            for item in list_items:
                if str(item) != 'labs':
                    markup.add(item)
            item = types.KeyboardButton('Вернуться в меню')
            markup.add(item)

            received_message_text = bot.send_message(
                message.chat.id, 'По какому предмету?', reply_markup=markup)
            bot.register_next_step_handler(received_message_text, change_book)

        elif message.text == 'Получить секретик':
            file = open('allowed_users.csv', 'r', encoding='utf-8')
            reader_der = csv.reader(file, delimiter=';')
            locked = True
            for row in reader_der:
                if str(row[0]) == str(message.from_user.id):
                    locked = False
            if locked == False:
                received_message = bot.send_message(
                    message.chat.id, 'Ура! У тебя есть доступ!\nНапиши,'
                                     ' что угодно. чтобы продолжить!')
                bot.register_next_step_handler(received_message, change_secret)
            else:
                markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
                button_change = types.KeyboardButton('/change')
                markup.add(button_change)
                received_message = bot.send_message(
                    message.chat.id, 'Блинб, у тебя нет доступа(\nНажми'
                                     ' "/change" чтобы продолжить', reply_markup=markup)
                bot.register_next_step_handler(received_message, change_option)

        elif message.text == 'Узнать расписание':
            sticker_really = open('data/stickers/REALLY.webp', 'rb')
            bot.send_sticker(message.chat.id, sticker_really)
            markup = types.ReplyKeyboardRemove()
            markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
            button_return_to_menu = types.KeyboardButton('Вернуться в меню')
            markup.add(button_return_to_menu)
            received_message = bot.send_message(
                message.chat.id, 'Введи мне номер своей группы, номер '
                                 'недели который тебе нужен и номер дня недели\n\n!!! '
                                 'Format: 6101-010302D 17 5 !!!', reply_markup=markup)
            bot.register_next_step_handler(received_message, send_schedule)

        elif message.text == 'Важные ссылки':
            important_links(message)

        elif message.text == 'News':
            if str(message.from_user.id) == '765103434':
                send_news(message)
            else:
                error(message)
        elif message.text == 'Вернуться в меню':
            change_option(message)

        else:
            error(message)


def error(message):
    """
    Handles errors that occur during the bot's operation.
    Sends an error message and suggests starting over.
    """
    sticker_idk = open('data/stickers/IDK.webp', 'rb')
    bot.send_sticker(message.chat.id, sticker_idk)
    markup = types.ReplyKeyboardRemove()
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    button_start = types.KeyboardButton('/start')
    markup.add(button_start)
    received_message_text = bot.send_message(
        message.chat.id, 'Блинб, я не знаю, что ответить(((\nНажми "/start" '
                         'чтобы продолжить', reply_markup=markup)
    bot.register_next_step_handler(received_message_text, welcome)


def open_file(way_to_file):
    """
    Checks if a file exists at the specified path.

    :param way_to_file: Path to the file.
    :return: True if the file exists, otherwise False.
    """
    return os.path.exists(way_to_file)


def important_links(message):
    """
    Sends important links to the user.
    """
    markup = types.ReplyKeyboardRemove()
    bot.send_message(message.chat.id, 'ИИК Приём - https://vk.com/iik.ssau.priem\nСтуд.'
                                      ' совет ИИК - https://vk.com/sciic\nРасписание ИИК '
                                      '- https://ssau.ru/rasp/faculty/492430598?course=1\nSSAU '
                                      '- https://ssau.ru\n', reply_markup=markup)
    print(
        f'Отправлено {message.text} -> Пользователь '
        f'{message.from_user.first_name} -> ID: {message.from_user.id}')
    change_option(message)


def send_schedule(message):
    """
    Sends the schedule to the user based on the provided data.

    :param message: Message from the user containing data for schedule retrieval.
    """
    if message.text == 'Вернуться в меню':
        change_option(message)
    else:
        if not os.path.isdir('AllGroupShedule'):
            gs.pars_all_group()
        try:
            num_group = message.text.split()[0]
            selected_week = message.text.split()[1]
            selected_weekday = message.text.split()[2]
            url_schedule = gs.find_schedule_url(
                num_group, selected_week, selected_weekday)
            schedule = gs.pars_shedule(url_schedule)
            bot.send_message(
                message.chat.id, schedule + f'\nURL: {url_schedule}')
            with open(f'data/work_with_group_id/{message.chat.id}.txt',
                      'w', encoding='utf-8') as file:
                file.write(num_group)
            print(
                f'Отправлено расписание {message.text} -> Пользователь '
                f'{message.from_user.first_name} -> ID: {message.from_user.id}')
            markup = types.ReplyKeyboardRemove()
            markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
            button_get_schedule = types.KeyboardButton('Узнать расписание сессии')
            button_return_to_menu = types.KeyboardButton('Вернуться в меню')
            markup.add(button_get_schedule, button_return_to_menu)
            received_message = bot.send_message(
                message.chat.id, 'Хочешь узнать что-то ещё?', reply_markup=markup)
            bot.register_next_step_handler(
                received_message, send_session_schedule)
        except:
            error(message)


def send_session_schedule(message):
    """
    Sends the session schedule to the user based on the group.

    :param message: Message from the user requesting the session schedule.
    """
    if message.text == 'Вернуться в меню':
        os.remove(f'data/work_with_group_id/{message.chat.id}.txt')
        change_option(message)
    else:
        try:
            num_group = ''
            with open(f'data/work_with_group_id/{message.chat.id}.txt',
                      'r', encoding='utf-8') as file:
                num_group = file.read()
            schedule = gs_session.pars_schedule_session(num_group)
            url_schedule = gs_session.find_schedule_session_url(num_group)
            markup = types.ReplyKeyboardRemove()
            bot.send_message(
                message.chat.id, schedule + f'\nURL: {url_schedule}', reply_markup=markup)
            os.remove(f'data/work_with_group_id/{message.chat.id}.txt')
            print(
                f'Отправлено расписание сессии {message.text} -> Пользователь '
                f'{message.from_user.first_name} -> ID: {message.from_user.id}')
            change_option(message)
        except:
            bot.send_message(
                message.chat.id, 'Возникла ошибка, возможно расписания сессии ещё нет(')


def send_news(message):
    """
    Sends news to all users.

    :param message: Message from the user initiating the news sending.
    """
    with open('data/info.txt', 'r', encoding='utf-8') as file:
        news = file.read()
    with open('users.csv', 'r', encoding='utf-8') as file:
        reader_der = csv.reader(file, delimiter=';')
        for row in reader_der:
            try:
                bot.send_message(row[2], news)
            except:
                pass
    change_option(message)


def change_lab_task(message):
    """
    Changes the lab assignment based on the user's selection.

    :param message: Message from the user with the selected assignment.
    """
    list_items = []
    list_files = []
    if message.text == 'Вернуться в меню':
        change_option(message)
    else:
        try:
            list_files = os.listdir(f'data/labs_book/labs/{message.text}')
            for files in list_files:
                list_items.append(files)
            markup = types.ReplyKeyboardRemove()
            markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
            for item in list_items:
                markup.add(item)
            button_return_to_menu = types.KeyboardButton('Вернуться в меню')
            markup.add(button_return_to_menu)
            received_message = bot.send_message(
                message.chat.id, 'Номер?', reply_markup=markup)
            bot.register_next_step_handler(received_message, send_pdf)
        except:
            error(message)


def change_book(message):
    """
    Changes the book based on the user's selection.

    :param message: Message from the user with the selected book.
    """
    list_items = []
    list_files = []
    markup = types.ReplyKeyboardRemove()
    if message.text == 'Вернуться в меню':
        change_option(message)
    else:
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
        try:
            list_files = os.listdir(f'data/labs_book/{message.text}')
            for files in list_files:
                list_items.append(files)
            markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
            for item in list_items:
                markup.add(item)
            button_return_to_menu = types.KeyboardButton('Вернуться в меню')
            markup.add(button_return_to_menu)
            received_message = bot.send_message(
                message.chat.id, 'Автор?', reply_markup=markup)
            bot.register_next_step_handler(received_message, send_pdf)
        except:
            error(message)


def change_secret(message):
    """
    Allows the user to choose a secret document.

    :param message: Message from the user requesting a secret document.
    """
    list_items = []
    for doc in os.listdir('secret'):
        list_items.append(doc)
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    for item in list_items:
        markup.add(item)
    button_return_to_menu = types.KeyboardButton('Вернуться в меню')
    markup.add(button_return_to_menu)
    received_message = bot.send_message(
        message.chat.id, 'Какой секретик нужен?', reply_markup=markup)
    bot.register_next_step_handler(received_message, send_secret)


def send_secret(message):
    """
    Sends the selected secret document to the user.

    :param message: Message from the user with the selected secret document.
    """
    if not message.text == 'Вернуться в меню':
        bot.send_message(message.chat.id, 'Отправляю!')
        file = open(f'secret/{message.text}', 'rb')
        bot.send_document(message.chat.id, file)
        print(
            f'Отправлен {message.text} -> Пользователь '
            f'{message.from_user.first_name} -> ID: {message.from_user.id}')
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
        button_change = types.KeyboardButton('/change')
        markup.add(button_change)
        received_message = bot.send_message(
            message.chat.id, 'Нажми "/change" чтобы продолжить', reply_markup=markup)
        bot.register_next_step_handler(received_message, change_option)
    else:
        change_option(message)


def send_pdf(message):
    """
    Sends a PDF document to the user based on their selection.

    :param message: Message from the user requesting a PDF document.
    """
    if message.text == 'Вернуться в меню':
        change_option(message)
    else:
        way_to_file = ''
        start_search = True
        while start_search:
            for subject in os.listdir('data/labs_book/'):
                way_to_file = f'data/labs_book/{subject}/{message.text}'
                if open_file(way_to_file):
                    start_search = False
                    break
            if start_search:
                while start_search:
                    for subject in os.listdir('data/labs_book/labs/'):
                        way_to_file = f'data/labs_book/labs/{subject}/{message.text}'
                        if open_file(way_to_file):
                            start_search = False
                            break
        needed_book = open(way_to_file, 'rb')
        bot.send_message(message.chat.id, 'Отправляю')
        bot.send_document(message.chat.id, needed_book)
        print(
            f'Отправлен {message.text} -> Пользователь '
            f'{message.from_user.first_name} -> ID: {message.from_user.id}')
        sticker_nya = open('data/stickers/NYA.webp', 'rb')
        bot.send_sticker(message.chat.id, sticker_nya)
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
        button_change = types.KeyboardButton('/change')
        markup.add(button_change)
        received_message = bot.send_message(
            message.chat.id, 'Нажми "/change" чтобы продолжить', reply_markup=markup)
        bot.register_next_step_handler(received_message, change_option)


while True:
    try:
        print("Eddie Start!")
        bot.polling(none_stop=True)
    except:
        print("Some problem, restart")
        time.sleep(15)
