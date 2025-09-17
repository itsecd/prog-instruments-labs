from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from aiogram_dialog import DialogManager, StartMode

from src.bot.states.main_menu import MainMenuStatesGroup


router = Router()


@router.message(Command("start"))
async def start_message_handler(message: Message, dialog_manager: DialogManager):
    await dialog_manager.start(MainMenuStatesGroup.main, mode=StartMode.RESET_STACK)


@router.message(Command("tasks"))
async def tasks_message_handler(message: Message, dialog_manager: DialogManager):
    await start_message_handler(message, dialog_manager)
