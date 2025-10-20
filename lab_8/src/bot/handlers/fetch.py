from aiogram import Router
from aiogram.types import Message
from aiogram.filters import Command
import aiohttp
import datetime


router = Router()


def prettify_user_info(user_info: dict) -> str:    
    prettified = ""

    site_admin = user_info["site_admin"]
    if site_admin:
        prettified += "✨ADMIN\n"

    prettified += f"<b>✏️Login (username):</b> <code>{user_info['login']}</code>\n"

    name = user_info["name"]
    if name is not None:
        prettified += f"<b>🤵Name:</b> <code>{name}</code>\n"
    
    bio = user_info["bio"]
    if bio is not None:
        prettified += f"<b>📒Bio:</b>\n<blockquote>{bio}</blockquote>\n\n"
    
    company = user_info["company"]
    if company is not None:
        prettified += f"<b>💼Company:</b> <code>{company}</code>\n"
    
    email = user_info["email"]
    if email is not None:
        prettified += f"<b>📧Email:</n> {email}"
    
    created_date = datetime.datetime.strptime(user_info['created_at'], "%Y-%m-%dT%H:%M:%SZ")
    created_date_string = datetime.datetime.strftime(created_date, "%Y/%m/%d")
    prettified += f"<b>🕒Created at:</b> <code>{created_date_string}</code>\n\n"

    prettified += f"<a href=\"{user_info['url']}\">🔗Link</a>"

    return prettified


async def get_github_user(username: str) -> dict | None:
    url = f"https://api.github.com/users/{username}"

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 404:
                return None

            j = await response.json()
            if isinstance(j, dict) and j.get("login") == username:
                return j

    raise RuntimeError(f"Someting went wrong: {j}")


@router.message(Command("github"))
async def github_command_handler(message: Message):
    args = message.text.split(" ", maxsplit=1)
    if len(args) < 2:
        await message.answer("Invalid command format")
        return

    username = args[-1]
    user_info = await get_github_user(username)
    if user_info is None:
        await message.answer("Invalid username")
        return
    
    await message.answer_photo(
        photo=user_info["avatar_url"],
        caption=prettify_user_info(user_info),
        parse_mode="HTML",
    )
