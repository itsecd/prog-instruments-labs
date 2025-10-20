from aiogram import Router
from aiogram.types import Message
from aiogram.filters import Command


router = Router()


@router.message(Command("start"))
async def start_command_handler(message: Message):
    await message.answer("Hello, it's menu!")
