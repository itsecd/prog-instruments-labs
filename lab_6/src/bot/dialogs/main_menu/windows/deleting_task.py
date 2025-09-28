from aiogram.types import CallbackQuery

from aiogram_dialog import Window, DialogManager
from aiogram_dialog.widgets.kbd import Button
from aiogram_dialog.widgets.text import Const, Format

import logging

from src.bot.states.main_menu import MainMenuStatesGroup
from src.utils import goto_state
from src.config import ui


logger = logging.getLogger(__name__)


async def _do_delete(
    callback: CallbackQuery,
    button: Button,
    dialog_manager: DialogManager,
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

    db = dialog_manager.middleware_data["db"]

    try:
        db.delete_task_by_idx(user_id, task_idx)
    except Exception as e:
        logger.exception("Error while deleting a task")
        await callback.answer(ui.errors.something_wrong)
        dialog_manager.switch_to(MainMenuStatesGroup.choosing_action)
        return
    
    logger.debug("Task deleted for user with id = %d", user_id)

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
