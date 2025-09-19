from aiogram.types import CallbackQuery

from aiogram_dialog import Window, DialogManager
from aiogram_dialog.widgets.kbd import Button
from aiogram_dialog.widgets.text import Const, List, Format

from src.bot.states.main_menu import MainMenuStatesGroup
from src.database.crud import (
    get_tasks,
    set_tasks_reversed,
    get_user,
)

from ..getters import tasks_getter

async def _do_add(callback: CallbackQuery, button: Button, dialog_manager: DialogManager):
    await dialog_manager.switch_to(MainMenuStatesGroup.adding_task)


async def _do_reverse(callback: CallbackQuery, button: Button, dialog_manager: DialogManager):
    user_id = callback.from_user.id
    user = get_user(user_id)
    set_tasks_reversed(user_id, not bool(user["tasks_reversed"]))

    await callback.answer(text="Reversed!")
    await dialog_manager.switch_to(MainMenuStatesGroup.choosing_action)


async def _do_select(callback: CallbackQuery, button: Button, dialog_manager: DialogManager):
    if len(get_tasks(callback.from_user.id)) == 0:
        await callback.answer("No tasks!")
        return

    await dialog_manager.switch_to(MainMenuStatesGroup.choosing_task)


window = Window(
    Const("Your tasks:"),
    List(
        Format("{item[0]}\. {item[1]}"),
        items="tasks",
    ),
    Button(Const("Add"), id="add_task", on_click=_do_add), # C
    Button(Const("Delete/Edit"), id="delete_or_edit_task", on_click=_do_select), # UD
    Button(Const("Reverse"), id="reverse_tasks", on_click=_do_reverse),
    state=MainMenuStatesGroup.choosing_action,
    getter=tasks_getter,
)
