from aiogram.types import CallbackQuery

from aiogram_dialog import Window, DialogManager
from aiogram_dialog.widgets.kbd import Button, Row
from aiogram_dialog.widgets.text import Const, Format

from src.bot.states.main_menu import MainMenuStatesGroup
from src.database.crud import get_tasks, delete_task_by_idx


async def do_back(callback: CallbackQuery, button: Button, dialog_manager: DialogManager):
    await dialog_manager.switch_to(MainMenuStatesGroup.task_choosen)


async def do_delete(callback: CallbackQuery, button: Button, dialog_manager: DialogManager):
    user_id = dialog_manager.event.from_user.id
    user_tasks = get_tasks(user_id)

    task_idx = dialog_manager.dialog_data.get("task_idx")
    if task_idx is None or task_idx >= len(user_tasks):
        await callback.answer("No task to delete!")
        return

    delete_task_by_idx(user_id, task_idx)
    await dialog_manager.switch_to(MainMenuStatesGroup.choosing_task)


window = Window(
    Format("Delete?"),
    Button(Const("Delete"), id="delete_task", on_click=do_delete),
    Button(Const("Back"), id="back", on_click=do_back),
    state=MainMenuStatesGroup.deleting_task,
)
