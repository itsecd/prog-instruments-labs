import telebot
from config import keys, TOKEN
from extensions import ConvertionException, CryptoConverter


bot = telebot.TeleBot(TOKEN)


@bot.message_handler(commands=['photo'])
def send_photo(message):
    with open('D:\study\Programming\PhotoMyBot.jpg', 'rb') as photo:
        bot.send_photo(chat_id=message.chat.id, photo=photo)


@bot.message_handler(commands=['chatinfo'])
def get_chat_info(message):
    chat = bot.get_chat(message.chat.id)
    chat_id = chat.id
    print(f"ID группы: {chat_id}")
    bot.reply_to(message, f"ID группы: {chat_id}")



@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, 'Привет! Я бот, который может помочь с конвертацией валют и не только. Введи /start, чтобы узнать, как я работаю.')


@bot.message_handler(commands=['start', 'help'])
def help(message: telebot.types.Message):
    text = 'Чтобы начать работу введите команду боту в следующем формате: ' \
           '\n<имя валюты> <в какие валюты перевести> ' \
           '\n<количество переводимой валюты>' \
           '\nУвидеть список доступных валют: /values'  \
           '\nКонвертация валют: /converts'\
            '\nИнтересные картинки:/photo '\
           '\nId чата: /chatinfo'
    bot.reply_to(message, text)


@bot.message_handler(commands=['values'])
def values(message: telebot.types.Message):
    text = 'Доступные валюты:'
    for key in keys.keys():
        text = '\n'.join((text, key, ))
    bot.reply_to(message, text)


@bot.message_handler(commands=['converts'])
def converts(message: telebot.types.Message):
    bot.reply_to(message,'Чтобы начать конвертацию введи интересующую тебя первую валюту, затем вторую и в каком эквиваленте.\n'
           '(Например: Биткоин рубль 3)\n'
           'К сожалению, можно использовать валюты только из этого списка /values')


@bot.message_handler(content_types=['text', ])
def convert(message: telebot.types.Message):
    try:
        values = message.text.split(' ')

        if len(values) != 3:
            raise ConvertionException('Недостаточно параметров. Введите: <имя валюты> <в какие валюты перевести> <количество переводимой валюты>')

        quote = values[0]
        bases = values[1:-1]
        amount = float(values[-1])

        total_bases = []
        for base in bases:
            total_base = CryptoConverter.convert(quote, base, amount)
            total_bases.append(total_base)

    except ConvertionException as e:
        bot.reply_to(message, f'Ошибка пользователя. \n{e}')

    except Exception as e:
        bot.reply_to(message, f'Не удалось обработать команду\n{e}')

    else:
        result = f'Цена {"1" if amount == 1 else f"{amount:.2f}"} {quote} в {base}:'
        for i, base in enumerate(bases):
            result += f'\n{base} - {total_bases[i]:.2f}'
        bot.send_message(message.chat.id, result)


bot.polling()