from aiogram.types import CallbackQuery

from aiogram_dialog import Window, DialogManager
from aiogram_dialog.widgets.kbd import Button, ScrollingGroup, Radio
from aiogram_dialog.widgets.text import Const, Format

from src.bot.states.main_menu import MainMenuStatesGroup

from .main_menu import get_tasks

async def do_back(callback: CallbackQuery, button: Button, dialog_manager: DialogManager):
    await dialog_manager.switch_to(MainMenuStatesGroup.main)


async def do_choose(callback: CallbackQuery, button: Button, dialog_manager: DialogManager):
    await dialog_manager.switch_to(MainMenuStatesGroup.task_choosen)


window = Window(
    Const("Select task:"),
    ScrollingGroup(
        Radio(
            Format("{item[0]}. {item[1]}"),
            Const(""),
            id="task_radio",
            item_id_getter=lambda x: str(x[0]),
            items="tasks",
            on_state_changed=MainMenuStatesGroup.task_choosen,
        ),
        id="tasks_group",
        width=1,
        height=5,
    ),
    Button(Const("Back"), id="back", on_click=do_back),
    state=MainMenuStatesGroup.selecting_task,
    getter=get_tasks,
)
