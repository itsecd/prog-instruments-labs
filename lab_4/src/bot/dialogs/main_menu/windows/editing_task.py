from aiogram.types import CallbackQuery, Message

from aiogram_dialog import Window, DialogManager
from aiogram_dialog.widgets.kbd import Button
from aiogram_dialog.widgets.text import Const
from aiogram_dialog.widgets.input import MessageInput

from src.bot.states.main_menu import MainMenuStatesGroup
from src.database.crud import update_task_by_idx

from src.config import ui


async def _do_back(callback: CallbackQuery, button: Button, dialog_manager: DialogManager):
    await dialog_manager.switch_to(MainMenuStatesGroup.task_choosen)


async def _edit_task(message: Message, _, dialog_manager: DialogManager):
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
    Button(Const(ui.buttons.back), id="back", on_click=_do_back),
    state=MainMenuStatesGroup.editing_task,
)
