from aiogram.types import CallbackQuery

from aiogram_dialog import Window, DialogManager
from aiogram_dialog.widgets.kbd import Button
from aiogram_dialog.widgets.text import Const, Format

from src.bot.states.main_menu import MainMenuStatesGroup
from src.database.crud import delete_task_by_idx
from src.utils import goto_state
from src.config import ui


async def _do_delete(
    callback: CallbackQuery,
    button: Button,
    dialog_manager: DialogManager
):
    """
    Click handler for deleting a task

    Args:
        message (Message): input message object
        button (Button): clicked button object
        dialog_manager (DialogManager): used dialog manager object
    """
    user_id = dialog_manager.event.from_user.id

    task_idx = dialog_manager.dialog_data.get("task_idx")

    try:
        delete_task_by_idx(user_id, task_idx)
    except Exception as e:
        # logging in other lab?
        await callback.answer(ui.errors.something_wrong)
        dialog_manager.switch_to(MainMenuStatesGroup.choosing_action)
        return

    await dialog_manager.switch_to(MainMenuStatesGroup.choosing_task)


window = Window(
    Format(ui.messages.ask_delete),
    Button(
        Const(ui.buttons.delete),
        id="delete_task",
        on_click=_do_delete
    ),
    Button(
        Const(ui.buttons.back),
        id="back",
        on_click=goto_state(MainMenuStatesGroup.task_choosen)
    ),
    state=MainMenuStatesGroup.deleting_task,
)
