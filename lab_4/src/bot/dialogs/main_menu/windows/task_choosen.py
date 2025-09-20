from aiogram_dialog import Window
from aiogram_dialog.widgets.kbd import Button, Row
from aiogram_dialog.widgets.text import Const, Format

from src.bot.states.main_menu import MainMenuStatesGroup
from src.utils import goto_state
from src.config import ui
from ..getters import task_getter


window = Window(
    Format(ui.messages.choose_action),
    Format("{task}"),
    Row(
        Button(
            Const("Edit"),
            id="edit_task",
            on_click=goto_state(MainMenuStatesGroup.editing_task)
        ),
        Button(
            Const("Delete"),
            id="delete_task",
            on_click=goto_state(MainMenuStatesGroup.deleting_task)
        ),
    ),
    Button(
        Const("Back"),
        id="back",
        on_click=goto_state(MainMenuStatesGroup.choosing_task)
    ),
    state=MainMenuStatesGroup.task_choosen,
    getter=task_getter,
)
