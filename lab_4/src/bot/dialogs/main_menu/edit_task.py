from aiogram.types import CallbackQuery, Message

from aiogram_dialog import Window, DialogManager
from aiogram_dialog.widgets.kbd import Button
from aiogram_dialog.widgets.text import Const
from aiogram_dialog.widgets.input import MessageInput

from src.bot.states.main_menu import MainMenuStatesGroup
from src.database.crud import update_task_by_idx

async def do_back(callback: CallbackQuery, button: Button, dialog_manager: DialogManager):
    await dialog_manager.switch_to(MainMenuStatesGroup.task_choosen)


async def edit_task(message: Message, _, dialog_manager: DialogManager):
    user_id = message.from_user.id
    task = message.text
    task_idx = dialog_manager.dialog_data.get("task_idx")

    update_task_by_idx(user_id, task_idx, task)

    await dialog_manager.switch_to(MainMenuStatesGroup.selecting_task)


window = Window(
    Const("Enter new task text:"),
    MessageInput(edit_task),
    Button(Const("Back"), id="back", on_click=do_back),
    state=MainMenuStatesGroup.editing_task,
)
