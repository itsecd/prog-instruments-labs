import asyncio
import os

from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession
import logging
from dotenv import load_dotenv
import handlers

logging.basicConfig(level=logging.INFO)
def register_routers(dp):
    dp.include_routers(handlers.router)


    
async def main() -> None:
    """
    Entry point
    """
    load_dotenv()
    session = AiohttpSession()
    bot = Bot(os.getenv("BOT_TOKEN"), session=session)
    dp = Dispatcher()
    register_routers(dp)
    try:
        await bot.delete_webhook()
        await  asyncio.create_task(dp.start_polling(bot))
        
    except Exception as ex:
        print(ex)


if __name__ == "__main__":
    asyncio.run(main())