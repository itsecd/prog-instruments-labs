import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram.fsm.storage.memory import MemoryStorage

from aiogram_dialog import setup_dialogs

from src.database.db import init_db

from src.bot.dialogs import DIALOGS
from src.bot.handlers import ROUTERS

from src.config import config

async def main():
    init_db()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    bot = Bot(
        token=config.bot_token,
        default=DefaultBotProperties(
            # parse_mode=ParseMode.MARKDOWN_V2,
        ),
    )

    dp = Dispatcher(storage=MemoryStorage())

    dp.include_routers(
        *ROUTERS,
        *DIALOGS,
    )
    setup_dialogs(dp)

    await bot.delete_webhook(drop_pending_updates=True)
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    init_db()
    from src.database.crud import get_user, create_user
    create_user(1)
    get_user(1)
    # try:
    #     asyncio.run(main())
    # except KeyboardInterrupt:
    #     logging.info("Exit Telegram Bot")
