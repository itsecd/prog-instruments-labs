import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage

from aiogram_dialog import setup_dialogs

from src.database.db import init_db
from src.bot.dialogs import DIALOGS
from src.bot.handlers import ROUTERS

from src.config import config


async def main():
    """Programm entry point"""
    init_db()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    bot = Bot(token=config.bot_token)

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
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Exit Telegram Bot")
