from aiogram.types import CallbackQuery, Message

from aiogram_dialog import Window, DialogManager
from aiogram_dialog.widgets.kbd import Button
from aiogram_dialog.widgets.text import Const
from aiogram_dialog.widgets.input import MessageInput

from src.bot.states.main_menu import MainMenuStatesGroup

from .main_menu import tasks


async def do_back(callback: CallbackQuery, button: Button, dialog_manager: DialogManager):
    await dialog_manager.switch_to(MainMenuStatesGroup.task_choosen)


async def edit_task(message: Message, _, dialog_manager: DialogManager):
    task = message.text

    tasks[message.from_user.id][dialog_manager.dialog_data["task_idx"]] = task

    await dialog_manager.switch_to(MainMenuStatesGroup.selecting_task)


window = Window(
    Const("Enter new task text:"),
    MessageInput(edit_task),
    Button(Const("Back"), id="back", on_click=do_back),
    state=MainMenuStatesGroup.editing_task,
)
