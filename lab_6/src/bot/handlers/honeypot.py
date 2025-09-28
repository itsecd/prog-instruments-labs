from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

import logging


logger = logging.getLogger(__name__)


router = Router()


async def honeypot_base_handler(message: Message):
    logger.warning(
        "Strange actions from user with id = %d, message = \"%s\"",
        message.from_user.id,
        message.text,
    )


@router.message(Command("test", "admin", "debug", "hack", "1337"))
async def honeypot_handler(message: Message):
    await honeypot_base_handler(message)
