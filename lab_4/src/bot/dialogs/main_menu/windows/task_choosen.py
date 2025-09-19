from aiogram.types import CallbackQuery

from aiogram_dialog import Window, DialogManager
from aiogram_dialog.widgets.kbd import Button, Row
from aiogram_dialog.widgets.text import Const, Format

from src.bot.states.main_menu import MainMenuStatesGroup

from src.config import ui

from ..getters import task_getter


async def _do_back(callback: CallbackQuery, button: Button, dialog_manager: DialogManager):
    await dialog_manager.switch_to(MainMenuStatesGroup.choosing_task)


async def _do_delete(callback: CallbackQuery, button: Button, dialog_manager: DialogManager):
    await dialog_manager.switch_to(MainMenuStatesGroup.deleting_task)


async def _do_edit(callback: CallbackQuery, button: Button, dialog_manager: DialogManager):
    await dialog_manager.switch_to(MainMenuStatesGroup.editing_task)


window = Window(
    Format(ui.messages.choose_action),
    Format("{task}"),
    Row(
        Button(Const("Edit"), id="edit_task", on_click=_do_edit),
        Button(Const("Delete"), id="delete_task", on_click=_do_delete),
    ),
    Button(Const("Back"), id="back", on_click=_do_back),
    state=MainMenuStatesGroup.task_choosen,
    getter=task_getter,
)
