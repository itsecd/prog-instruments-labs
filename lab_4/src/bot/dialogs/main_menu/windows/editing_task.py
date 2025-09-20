from aiogram.types import Message

from aiogram_dialog import Window, DialogManager
from aiogram_dialog.widgets.kbd import Button
from aiogram_dialog.widgets.text import Const
from aiogram_dialog.widgets.input import MessageInput

from src.bot.states.main_menu import MainMenuStatesGroup
from src.database.crud import update_task_by_idx
from src.utils import goto_state
from src.config import ui


async def _edit_task(
    message: Message,
    button: Button,
    dialog_manager: DialogManager
):
    """
    Input message handler for editing a task

    Args:
        message (Message): input message object
        button (Button): clicked button object
        dialog_manager (DialogManager): used dialog manager object
    """
    user_id = message.from_user.id
    task = message.text
    task_idx = dialog_manager.dialog_data.get("task_idx")

    try:
        update_task_by_idx(user_id, task_idx, task)
    except Exception as e:
        # logging in other lab?
        await message.answer(ui.messages.something_wrong)
        await dialog_manager.switch_to(MainMenuStatesGroup.choosing_task)
        return

    await dialog_manager.switch_to(MainMenuStatesGroup.choosing_task)


window = Window(
    Const(ui.messages.enter_new_task),
    MessageInput(_edit_task),
    Button(
        Const(ui.buttons.back),
        id="back",
        on_click=goto_state(MainMenuStatesGroup.task_choosen)
    ),
    state=MainMenuStatesGroup.editing_task,
)
