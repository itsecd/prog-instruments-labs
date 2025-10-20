from aiogram import Router
from aiogram.types import Message
from aiogram.filters import Command
import logging

from src.utils.github import get_github_user, prettify_user_info


router = Router()


@router.message(Command("github"))
async def github_command_handler(message: Message):
    args = message.text.split(" ", maxsplit=1)
    if len(args) < 2:
        logging.error("Invalid command format: %s", message.text)
        await message.answer("Invalid command format. Good: /github <username>")
        return

    username = args[-1]
    user_info = await get_github_user(username)
    if user_info is None:
        logging.error("Fetching invalid username: %s", username)
        await message.answer("Invalid username")
        return
    
    await message.answer_photo(
        photo=user_info["avatar_url"],
        caption=prettify_user_info(user_info),
        parse_mode="HTML",
    )
