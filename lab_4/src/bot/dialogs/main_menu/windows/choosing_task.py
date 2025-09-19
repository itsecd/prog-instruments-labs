from aiogram.types import CallbackQuery

from aiogram_dialog import Window, DialogManager
from aiogram_dialog.widgets.kbd import Button, ScrollingGroup, Radio
from aiogram_dialog.widgets.text import Const, Format

from src.bot.states.main_menu import MainMenuStatesGroup
from src.database.crud import get_tasks_count, get_user
from src.utils import goto_state
from src.config import ui
from ..getters import tasks_getter


async def _do_choose(
    callback: CallbackQuery,
    button: Button,
    dialog_manager: DialogManager,
    item_id: str
):
    user_id = callback.from_user.id

    try:
        tasks_count = get_tasks_count(user_id)
        tasks_reversed = bool(get_user(user_id)["tasks_reversed"])
    except Exception as e:
        # logging in other lab?
        callback.answer(ui.errors.something_wrong)
        await dialog_manager.switch_to(MainMenuStatesGroup.choosing_action)
        return

    task_idx = int(item_id) - 1
    if tasks_reversed:
        task_idx = tasks_count - task_idx - 1

    dialog_manager.dialog_data["task_idx"] = task_idx
    await dialog_manager.switch_to(MainMenuStatesGroup.task_choosen)


window = Window(
    Const(ui.messages.choose_task),
    ScrollingGroup(
        Radio(
            Format(ui.formats.tasks),
            Format(ui.formats.tasks),
            id="task_radio",
            item_id_getter=lambda x: str(x[0]),
            items="tasks_for_buttons",
            on_click=_do_choose,
        ),
        id="tasks_group",
        width=1,
        height=5,
    ),
    Button(
        Const(ui.buttons.back),
        id="back",
        on_click=goto_state(MainMenuStatesGroup.choosing_action)
    ),
    state=MainMenuStatesGroup.choosing_task,
    getter=tasks_getter,
)
