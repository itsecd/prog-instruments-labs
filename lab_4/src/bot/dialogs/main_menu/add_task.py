from aiogram.types import CallbackQuery, Message

from aiogram_dialog import Window, DialogManager
from aiogram_dialog.widgets.input import MessageInput
from aiogram_dialog.widgets.kbd import Button
from aiogram_dialog.widgets.text import Const

from src.bot.states.main_menu import MainMenuStatesGroup


async def do_back(callback: CallbackQuery, button: Button, dialog_manager: DialogManager):
    await dialog_manager.switch_to(MainMenuStatesGroup.main)


async def save_task(message: Message, _, dialog_manager: DialogManager):
    task = message.text

    from .main_menu import tasks
    user_tasks = tasks.get(message.from_user.id, [])
    if user_tasks == []:
        tasks[message.from_user.id] = [task]
    else:
        tasks[message.from_user.id].append(task) 

    await dialog_manager.switch_to(MainMenuStatesGroup.main)


window = Window(
    Const("Enter task:"),
    MessageInput(save_task),
    Button(Const("Back"), id="back", on_click=do_back),
    state=MainMenuStatesGroup.adding_task,
)
