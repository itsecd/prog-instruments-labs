from aiogram.types import CallbackQuery, Message

from aiogram_dialog import Window, DialogManager
from aiogram_dialog.widgets.input import MessageInput
from aiogram_dialog.widgets.kbd import Button
from aiogram_dialog.widgets.text import Const

from src.bot.states.main_menu import MainMenuStatesGroup
from src.database.crud import add_task


async def do_back(callback: CallbackQuery, button: Button, dialog_manager: DialogManager):
    await dialog_manager.switch_to(MainMenuStatesGroup.main)


async def save_task(message: Message, _, dialog_manager: DialogManager):
    task = message.text

    add_task(message.from_user.id, task)

    await dialog_manager.switch_to(MainMenuStatesGroup.main)


window = Window(
    Const("Enter task:"),
    MessageInput(save_task),
    Button(Const("Back"), id="back", on_click=do_back),
    state=MainMenuStatesGroup.adding_task,
)
