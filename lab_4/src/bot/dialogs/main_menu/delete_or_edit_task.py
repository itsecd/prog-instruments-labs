from aiogram.types import CallbackQuery

from aiogram_dialog import Window, DialogManager
from aiogram_dialog.widgets.kbd import Button, Row
from aiogram_dialog.widgets.text import Const, Format

from src.bot.states.main_menu import MainMenuStatesGroup

from .main_menu import tasks


async def do_back(callback: CallbackQuery, button: Button, dialog_manager: DialogManager):
    await dialog_manager.switch_to(MainMenuStatesGroup.selecting_task)


async def get_task(dialog_manager: DialogManager, **kwargs):
    user_id = dialog_manager.event.from_user.id
    user_tasks = tasks.get(user_id, [])

    task_idx = dialog_manager.dialog_data.get("task_idx")
    if task_idx is None or task_idx >= len(user_tasks):
        return {"task": "No task selected"}

    return {"task": user_tasks[task_idx]}


async def do_delete(callback: CallbackQuery, button: Button, dialog_manager: DialogManager):
    await dialog_manager.switch_to(MainMenuStatesGroup.deleting_task)


async def do_edit(callback: CallbackQuery, button: Button, dialog_manager: DialogManager):
    await dialog_manager.switch_to(MainMenuStatesGroup.editing_task)


window = Window(
    Format("Selected action fot task:\n{task}"),
    Row(
        Button(Const("Edit"), id="edit_task", on_click=do_edit),
        Button(Const("Delete"), id="delete_task", on_click=do_delete),
    ),
    Button(Const("Back"), id="back", on_click=do_back),
    state=MainMenuStatesGroup.task_choosen,
    getter=get_task,
)
