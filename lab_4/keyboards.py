from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

def user_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(keyboard=[
        [KeyboardButton(text='Лаба открыта?')]
    ], resize_keyboard=True, one_time_keyboard=False)
    
    
def admin_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(keyboard=[
        [KeyboardButton(text='Лаба открыта?')],
        [KeyboardButton(text='Открыть'), KeyboardButton(text='Закрыть')]
    ], resize_keyboard=True, one_time_keyboard=False)