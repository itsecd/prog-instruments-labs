from aiogram.types import CallbackQuery

from aiogram_dialog import Window, DialogManager
from aiogram_dialog.widgets.kbd import Button, ScrollingGroup, Radio
from aiogram_dialog.widgets.text import Const, Format

from src.bot.states.main_menu import MainMenuStatesGroup

from ..getters import tasks_getter


async def _do_back(callback: CallbackQuery, button: Button, dialog_manager: DialogManager):
    await dialog_manager.switch_to(MainMenuStatesGroup.choosing_action)


async def _do_choose(callback: CallbackQuery, button: Button, dialog_manager: DialogManager, item_id: str):
    task_idx = int(item_id) - 1
    dialog_manager.dialog_data["task_idx"] = task_idx
    await dialog_manager.switch_to(MainMenuStatesGroup.task_choosen)


window = Window(
    Const("Select task:"),
    ScrollingGroup(
        Radio(
            Format("{item[0]}. {item[1]}"),
            Format("{item[0]}. {item[1]}"),
            id="task_radio",
            item_id_getter=lambda x: str(x[0]),
            items="tasks",
            on_click=_do_choose,
        ),
        id="tasks_group",
        width=1,
        height=5,
    ),
    Button(Const("Back"), id="back", on_click=_do_back),
    state=MainMenuStatesGroup.choosing_task,
    getter=tasks_getter,
)
