import asyncio
import logging

from aiogram import Bot, Dispatcher

from src.bot.handlers import routers

from src.config import config


logger = logging.getLogger(__name__)


async def main():
    """Programm entry point"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    logger.info("Program starting")

    bot = Bot(token=config.tg_bot_token)

    dp = Dispatcher()

    dp.include_routers(
        *routers,
    )

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
