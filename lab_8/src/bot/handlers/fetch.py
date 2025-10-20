from aiogram import Router
from aiogram.types import Message
from aiogram.filters import Command

import aiohttp


router = Router()


async def get_github_user(username: str) -> dict | None:
    url = f"https://api.github.com/users/{username}"

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 404:
                return None

            j = response.json()

            return {
                j["id"],
                j["login"],
                j["avatar_url"],
                j["created_at"],
            }


@router.message(Command("github"))
async def github_command_handler(message: Message):
    args = message.text.split(" ", maxsplit=1)
    if len(args) < 2:
        await message.answer("Invalid command format")
        return

    username = args[-1]
