from aiogram.types import CallbackQuery

from aiogram_dialog import Window, DialogManager
from aiogram_dialog.widgets.kbd import Button
from aiogram_dialog.widgets.text import Const, List, Format

from src.bot.states.main_menu import MainMenuStatesGroup
from src.database.crud import get_tasks_count, set_tasks_reversed, get_user
from src.utils import goto_state
from src.config import ui
from ..getters import tasks_getter


async def _do_reverse(
    callback: CallbackQuery,
    button: Button,
    dialog_manager: DialogManager
):
    user_id = callback.from_user.id

    try:
        user = get_user(user_id)
        set_tasks_reversed(user_id, not bool(user["tasks_reversed"]))
    except Exception as e:
        # logging in other lab?
        await callback.answer(ui.errors.something_wrong)
    else:
        await callback.answer(text="Reversed!")

    await dialog_manager.switch_to(MainMenuStatesGroup.choosing_action)


async def _do_select(
    callback: CallbackQuery,
    button: Button,
    dialog_manager: DialogManager
):
    try:
        tasks_count = get_tasks_count(callback.from_user.id)
    except Exception as e:
        # logging in other lab?
        await callback.answer(ui.errors.something_wrong)
        return

    if tasks_count == 0:
        await callback.answer("No tasks!")
        return

    await dialog_manager.switch_to(MainMenuStatesGroup.choosing_task)


window = Window(
    Const(ui.messages.your_tasks),
    List(
        Format(ui.formats.tasks),
        items="tasks",
    ),
    Button(
        Const(ui.buttons.add),
        id="add_task",
        on_click=goto_state(MainMenuStatesGroup.adding_task)
    ),
    Button(
        Const(ui.buttons.edit_or_delete),
        id="select_task",
        on_click=_do_select
    ),
    Button(
        Const(ui.buttons.reverse),
        id="reverse_tasks",
        on_click=_do_reverse
    ),
    state=MainMenuStatesGroup.choosing_action,
    getter=tasks_getter,
)
