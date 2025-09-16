from aiogram.types import ReplyKeyboardMarkup, InlineKeyboardMarkup, InlineKeyboardButton, KeyboardButton, ReplyKeyboardRemove
from aiogram.utils.keyboard import ReplyKeyboardBuilder
from data import data

startMarkup = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text='Да! хочу', callback_data="startYes")],
    [InlineKeyboardButton(text='Нет( попозже', callback_data="startNo")]
])

actionMarkup = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text='просмотреть таск-лист🔼', callback_data="getTask")],
    [InlineKeyboardButton(text='добавить задание➕', callback_data="addTask")],
    [InlineKeyboardButton(text='"вычеркнуть" задание❌', callback_data="delTask")],
    [InlineKeyboardButton(text='Редактировать🗒', callback_data="editTask")],
])

adderMarkup = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text='добавить еще задание➕', callback_data="addanother")],
    [InlineKeyboardButton(text='достаточно✅', callback_data="deleditTaskBack")]
])

getterMarkup = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text='скрыть таск-лист🔽', callback_data="hideTask")],
    [InlineKeyboardButton(text='добавить задание➕', callback_data="addTask")],
    [InlineKeyboardButton(text='"вычеркнуть" задание❌', callback_data="delTask")],
    [InlineKeyboardButton(text='Редактировать🗒', callback_data="editTask")]
])

dellerMarkup = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text='выбрать задания для удаления❌', callback_data="TaskDeller")],
    [InlineKeyboardButton(text='очистить таск-лист🗑', callback_data="clearTask")],
    [InlineKeyboardButton(text='назад↪', callback_data="deleditTaskBack")]
])

def taskskbbuilder(userid, finalbutton):
    builder = ReplyKeyboardBuilder()
    for i in range(1, data.getCurNumOfTask(userid)):
           builder.add(KeyboardButton(text=str(i)))
    builder.add(KeyboardButton(text = f"{finalbutton}"))
    builder.adjust(4)
    return builder.as_markup(resize_keyboard=True)

dellerMarkupback = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text='вернуться↪', callback_data="deleditTaskBack")]
])

editorMarkup = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text='выбрать задания для редактирования🗒', callback_data="TaskEditor")],
    [InlineKeyboardButton(text='перевернуть таск-лист🔄', callback_data="reverseTask")],
    [InlineKeyboardButton(text='назад↪', callback_data="deleditTaskBack")]
])
