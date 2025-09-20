from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from aiogram_dialog import DialogManager, StartMode

from contextlib import suppress
from sqlite3 import IntegrityError

from src.bot.states.main_menu import MainMenuStatesGroup
from src.database.crud import create_user


router = Router()


@router.message(Command("start"))
async def start_message_handler(
    message: Message,
    dialog_manager: DialogManager
):
    """Handler for /start command"""
    with suppress(IntegrityError):
        create_user(message.from_user.id)

    await dialog_manager.start(
        MainMenuStatesGroup.choosing_action,
        mode=StartMode.RESET_STACK,
    )


@router.message(Command("tasks"))
async def tasks_message_handler(
    message: Message,
    dialog_manager: DialogManager
):
    """Handler for /tasks command"""
    await start_message_handler(message, dialog_manager)
