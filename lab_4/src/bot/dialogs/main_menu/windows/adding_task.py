from aiogram.types import Message

from aiogram_dialog import Window, DialogManager
from aiogram_dialog.widgets.input import MessageInput
from aiogram_dialog.widgets.kbd import Button
from aiogram_dialog.widgets.text import Const

from src.bot.states.main_menu import MainMenuStatesGroup
from src.database.crud import add_task
from src.utils import goto_state
from src.config import ui


async def _save_task(message: Message, _, dialog_manager: DialogManager):
    task = message.text

    try:
        add_task(message.from_user.id, task)
    except Exception as e:
        # logging in other labs?
        await message.answer(ui.errors.something_wrong)

    await dialog_manager.switch_to(MainMenuStatesGroup.choosing_action)


window = Window(
    Const(ui.messages.enter_task),
    MessageInput(_save_task),
    Button(Const(ui.buttons.back), id="back", on_click=goto_state(MainMenuStatesGroup.choosing_action)),
    state=MainMenuStatesGroup.adding_task,
)
