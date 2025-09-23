import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage

from aiogram_dialog import setup_dialogs

from src.database.db import init_db
from src.bot.dialogs import DIALOGS
from src.bot.handlers import ROUTERS

from src.config import config


logger = logging.getLogger(__name__)


async def main():
    """Programm entry point"""
    logging.basicConfig(
        level=config.logging_level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    logger.info("Program starting")

    logger.info("Database initialization")
    init_db()

    bot = Bot(token=config.bot_token)

    dp = Dispatcher(storage=MemoryStorage())

    dp.include_routers(
        *ROUTERS,
        *DIALOGS,
    )

    logger.info("Setting up aiogram dialogs for dispatcher...")
    setup_dialogs(dp)

    logger.info("Deleting webhooks")
    await bot.delete_webhook(drop_pending_updates=True)
    
    logger.info("Dispatcher starting polling...")
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()
        logger.info("Bot session closed")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Exit Telegram Bot")
