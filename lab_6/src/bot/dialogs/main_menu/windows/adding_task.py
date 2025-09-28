from aiogram.types import Message

from aiogram_dialog import Window, DialogManager
from aiogram_dialog.widgets.input import MessageInput
from aiogram_dialog.widgets.kbd import Button
from aiogram_dialog.widgets.text import Const

import logging

from src.bot.states.main_menu import MainMenuStatesGroup
from src.utils import goto_state
from src.config import ui


logger = logging.getLogger(__name__)


async def _save_task(
    message: Message,
    button: Button,
    dialog_manager: DialogManager,
):
    """
    Input message handler for saving a task

    Args:
        message (Message): input message object
        button (Button): clicked button object
        dialog_manager (DialogManager): used dialog manager object
    """
    task = message.text

    db = dialog_manager.middleware_data["db"]

    try:
        db.add_task(message.from_user.id, task)
    except Exception:
        logger.exception("Error while saving a task")
        await message.answer(ui.errors.something_wrong)
    
    logger.debug(
        "Task \"%s\" saved for user with id = %d",
        task,
        message.from_user.id
    )

    await dialog_manager.switch_to(MainMenuStatesGroup.choosing_action)


window = Window(
    Const(ui.messages.enter_task),
    MessageInput(_save_task),
    Button(Const(ui.buttons.back), id="back", on_click=goto_state(MainMenuStatesGroup.choosing_action)),
    state=MainMenuStatesGroup.adding_task,
)
