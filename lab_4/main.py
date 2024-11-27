import asyncio
import os

from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession
import logging
from dotenv import load_dotenv
import handlers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    filename="py_log.log",
    filemode="w",
)


def register_routers(dp):
    dp.include_routers(handlers.router)


async def main() -> None:
    """
    Entry point
    """
    logging.info("Загрузка переменных из файла в окружение")
    load_dotenv()
    session = AiohttpSession()
    bot = Bot(os.getenv("BOT_TOKEN"), session=session)
    dp = Dispatcher()
    register_routers(dp)
    try:
        await bot.delete_webhook()
        await asyncio.create_task(dp.start_polling(bot))

    except Exception as ex:
        print(ex)


if __name__ == "__main__":
    logging.info("Старт процесса")
    asyncio.run(main())
