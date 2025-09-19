from aiogram.types import CallbackQuery, Message

from aiogram_dialog import Window, DialogManager
from aiogram_dialog.widgets.input import MessageInput
from aiogram_dialog.widgets.kbd import Button
from aiogram_dialog.widgets.text import Const

from src.bot.states.main_menu import MainMenuStatesGroup
from src.database.crud import add_task


async def _do_back(callback: CallbackQuery, button: Button, dialog_manager: DialogManager):
    await dialog_manager.switch_to(MainMenuStatesGroup.choosing_action)


async def _save_task(message: Message, _, dialog_manager: DialogManager):
    task = message.text

    add_task(message.from_user.id, task)

    await dialog_manager.switch_to(MainMenuStatesGroup.choosing_action)


window = Window(
    Const("Enter task:"),
    MessageInput(_save_task),
    Button(Const("Back"), id="back", on_click=_do_back),
    state=MainMenuStatesGroup.adding_task,
)
