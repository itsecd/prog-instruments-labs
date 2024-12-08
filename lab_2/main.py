from datetime import date
from aiogram.utils.exceptions import MessageNotModified
from aiogram import Bot, Dispatcher, executor, types
from config import *
from utils import profile_text, ref_text, pay_text, get_all_mamonts, add_mamont
from buttons import *

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)


@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    mamont_id = message.from_user.id
    if " " in message.text:
        referrer = message.text.split()[1]
        try:
            referrer = int(referrer)
            if mamont_id != referrer and mamont_id not in get_all_mamonts():
                add_mamont(referrer, mamont_id, date.today())
        except ValueError:
            pass
    await bot.send_photo(chat_id=message.chat.id,
                         photo=open("images/start.png", "rb"),
                         caption=main_text,
                         reply_markup=markup_start)


@dp.message_handler(commands=['воркер'])
async def send_worker(message: types.Message):
    await message.answer(worker_text, reply_markup=markup_worker)


@dp.callback_query_handler(lambda call: call.data == 'worker_ref')
async def worker_ref(call: types.CallbackQuery):
    await bot.answer_callback_query(call.id)
    try:
        await bot.edit_message_text(ref_text(call.from_user.id),
                                    reply_markup=markup_worker,
                                    inline_message_id=call.inline_message_id,
                                    message_id=call.message.message_id,
                                    chat_id=call.message.chat.id)
    except MessageNotModified:
        pass


@dp.callback_query_handler(lambda call: call.data == 'worker_buy')
async def worker_buy(call: types.CallbackQuery):
    await bot.answer_callback_query(call.id)
    await bot.send_message(text=worker_buy_text, chat_id=call.message.chat.id)


@dp.message_handler(content_types=['text'])
async def worker_add_orders(message: types.Message):
    if 'накрутить' in message.text.lower():
        try:
            count = message.text.split(' ')[1]
            await bot.send_message(text=profile_text(message.from_user.id, count), chat_id=message.chat.id)
        except IndexError:
            pass

@dp.message_handler(commands=['help'])
async def send_help(message: types.Message):
    help_text = (
        "Доступные команды:\n"
        "/start - Начать взаимодействие с ботом\n"
        "/воркер - Получить информацию о воркере\n"
        "/help - Показать это сообщение\n"
        "Нажмите на кнопки в меню, чтобы получить доступ к другим функциям."
    )
    await message.answer(help_text, reply_markup=markup_main)


@dp.callback_query_handler(lambda call: call.data == 'back_main')
async def main_layout(call: types.CallbackQuery):
    await bot.answer_callback_query(call.id)
    await bot.edit_message_text(back_main,
                                reply_markup=markup_main,
                                inline_message_id=call.inline_message_id,
                                message_id=call.message.message_id,
                                chat_id=call.message.chat.id)


@dp.callback_query_handler(lambda call: 'links' in call.data)
async def links(call: types.CallbackQuery):
    await bot.answer_callback_query(call.id)
    if int(call.data[-1]) == 0:
        await bot.send_message(chat_id=call.message.chat.id, text=links_text, reply_markup=markup_feedback)
    else:
        try:
            await bot.edit_message_text(links_text,
                                        reply_markup=markup_feedback,
                                        inline_message_id=call.inline_message_id,
                                        message_id=call.message.message_id,
                                        chat_id=call.message.chat.id)
        except MessageNotModified:
            pass


@dp.callback_query_handler(lambda call: 'feedback' in call.data)
async def feedback(call: types.CallbackQuery):
    await bot.answer_callback_query(call.id)
    try:
        await bot.edit_message_text(feedback_list[int(call.data[-1])],
                                    reply_markup=markup_feedback,
                                    inline_message_id=call.inline_message_id,
                                    message_id=call.message.message_id,
                                    chat_id=call.message.chat.id)
    except MessageNotModified:
        pass


@dp.callback_query_handler(lambda call: 'buy_stuff' in call.data)
async def buy_process(call: types.CallbackQuery):
    await bot.answer_callback_query(call.id)
    if int(call.data[-1]) == 0:
        await bot.send_message(chat_id=call.message.chat.id, text=buy_text, reply_markup=markup_city)
    else:
        await bot.edit_message_text(buy_text,
                                    reply_markup=markup_city,
                                    inline_message_id=call.inline_message_id,
                                    message_id=call.message.message_id,
                                    chat_id=call.message.chat.id)


@dp.callback_query_handler(lambda call: 'vacancies' in call.data)
async def vacancies(call: types.CallbackQuery):
    await bot.answer_callback_query(call.id)
    if int(call.data[-1]) == 0:
        await bot.send_message(chat_id=call.message.chat.id, text=vacancies_text, reply_markup=markup_main)
    else:
        try:
            await bot.edit_message_text(vacancies_text,
                                        reply_markup=markup_main,
                                        inline_message_id=call.inline_message_id,
                                        message_id=call.message.message_id,
                                        chat_id=call.message.chat.id)
        except MessageNotModified:
            pass


@dp.callback_query_handler(lambda call: 'promo' in call.data)
async def promo(call: types.CallbackQuery):
    await bot.answer_callback_query(call.id)
    if int(call.data[-1]) == 0:
        await bot.send_message(chat_id=call.message.chat.id, text=ref_text(call.from_user.id), reply_markup=markup_main)
    else:
        try:
            await bot.edit_message_text(ref_text(call.from_user.id),
                                        reply_markup=markup_main,
                                        inline_message_id=call.inline_message_id,
                                        message_id=call.message.message_id,
                                        chat_id=call.message.chat.id)
        except MessageNotModified:
            pass


@dp.callback_query_handler(lambda call: 'important' in call.data)
async def important(call: types.CallbackQuery):
    await bot.answer_callback_query(call.id)
    if int(call.data[-1]) == 0:
        await bot.send_message(chat_id=call.message.chat.id, text=important_text, reply_markup=markup_main)
    else:
        try:
            await bot.edit_message_text(important_text,
                                        reply_markup=markup_main,
                                        inline_message_id=call.inline_message_id,
                                        message_id=call.message.message_id,
                                        chat_id=call.message.chat.id)
        except MessageNotModified:
            pass


@dp.callback_query_handler(lambda call: 'profile' in call.data)
async def profile(call: types.CallbackQuery):
    await bot.answer_callback_query(call.id)
    if int(call.data[-1]) == 0:
        await bot.send_message(chat_id=call.message.chat.id, text=profile_text(call.from_user.id),
                               reply_markup=markup_main)
    else:
        try:
            await bot.edit_message_text(profile_text(call.from_user.id),
                                        reply_markup=markup_main,
                                        inline_message_id=call.inline_message_id,
                                        message_id=call.message.message_id,
                                        chat_id=call.message.chat.id)
        except MessageNotModified:
            pass


@dp.callback_query_handler(lambda call: call.data == 'city_moscow')
async def moscow_layout(call: types.CallbackQuery):
    await bot.answer_callback_query(call.id)
    await bot.edit_message_text("Выберите район города Москва:",
                                reply_markup=markup_moscow,
                                inline_message_id=call.inline_message_id,
                                message_id=call.message.message_id,
                                chat_id=call.message.chat.id)


@dp.callback_query_handler(lambda call: call.data == 'city_saintP')
async def saint_p_layout(call: types.CallbackQuery):
    await bot.answer_callback_query(call.id)
    await bot.edit_message_text("Выберите район города Санкт-Петербург:",
                                reply_markup=markup_saintP,
                                inline_message_id=call.inline_message_id,
                                message_id=call.message.message_id,
                                chat_id=call.message.chat.id)


@dp.callback_query_handler(lambda call: 'city_default' in call.data)
async def city_default_layout(call: types.CallbackQuery):
    await bot.answer_callback_query(call.id)
    await bot.edit_message_text("Выберите местоположение",
                                reply_markup=markup_city_default,
                                inline_message_id=call.inline_message_id,
                                message_id=call.message.message_id,
                                chat_id=call.message.chat.id)


@dp.callback_query_handler(lambda call: 'price_list' in call.data)
async def price_list(call: types.CallbackQuery):
    await bot.answer_callback_query(call.id)
    await bot.edit_message_text(price_list_text,
                                reply_markup=markup_price,
                                inline_message_id=call.inline_message_id,
                                message_id=call.message.message_id,
                                chat_id=call.message.chat.id)


@dp.callback_query_handler(lambda call: call.data == 'price_back')
async def price_list1(call: types.CallbackQuery):
    await bot.answer_callback_query(call.id)
    await bot.send_message(chat_id=call.message.chat.id, text=price_list_text, reply_markup=markup_price)


@dp.callback_query_handler(lambda call: call.data == 's1')
async def s1(call: types.CallbackQuery):
    await bot.answer_callback_query(call.id)
    pic = open("images/s1.jpg", "rb")
    await bot.send_photo(chat_id=call.message.chat.id, photo=pic, caption=s1_text, reply_markup=markup_s1)


@dp.callback_query_handler(lambda call: call.data == 's2')
async def s2(call: types.CallbackQuery):
    await bot.answer_callback_query(call.id)
    pic = open("images/s2.jpg", "rb")
    await bot.send_photo(chat_id=call.message.chat.id, photo=pic, caption=s2_text, reply_markup=markup_s2)


@dp.callback_query_handler(lambda call: call.data == 's3')
async def s3(call: types.CallbackQuery):
    await bot.answer_callback_query(call.id)
    pic = open("images/s3.jpg", "rb")
    await bot.send_photo(chat_id=call.message.chat.id, photo=pic, caption=s3_text, reply_markup=markup_s3)


@dp.callback_query_handler(lambda call: call.data == 's4')
async def s4(call: types.CallbackQuery):
    await bot.answer_callback_query(call.id)
    pic = open("images/s4.jpg", "rb")
    await bot.send_photo(chat_id=call.message.chat.id, photo=pic, caption=s4_text, reply_markup=markup_s4)


@dp.callback_query_handler(lambda call: call.data == 's5')
async def s5(call: types.CallbackQuery):
    await bot.answer_callback_query(call.id)
    pic = open("images/s5.jpg", "rb")
    await bot.send_photo(chat_id=call.message.chat.id, photo=pic, caption=s5_text, reply_markup=markup_s5)


@dp.callback_query_handler(lambda call: call.data == 's6')
async def s6(call: types.CallbackQuery):
    await bot.answer_callback_query(call.id)
    pic = open("images/s6.jpg", "rb")
    await bot.send_photo(chat_id=call.message.chat.id, photo=pic, caption=s6_text, reply_markup=markup_s6)


@dp.callback_query_handler(lambda call: call.data == 's7')
async def s7(call: types.CallbackQuery):
    await bot.answer_callback_query(call.id)
    pic = open("images/s7.jpg", "rb")
    await bot.send_photo(chat_id=call.message.chat.id, photo=pic, caption=s7_text, reply_markup=markup_s7)


@dp.callback_query_handler(lambda call: call.data == 's8')
async def s8(call: types.CallbackQuery):
    await bot.answer_callback_query(call.id)
    pic = open("images/s8.jpg", "rb")
    await bot.send_photo(chat_id=call.message.chat.id, photo=pic, caption=s8_text, reply_markup=markup_s8)


@dp.callback_query_handler(lambda call: 'pay' in call.data)
async def pay(call: types.CallbackQuery):
    await bot.answer_callback_query(call.id)
    await bot.send_message(chat_id=call.message.chat.id,
                           text=pay_text(prices[call.data[-2]][int(call.data[-1])]),
                           reply_markup=markup_pay)


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)