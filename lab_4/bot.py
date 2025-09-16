import asyncio
from aiogram import Bot, Dispatcher

#local imports
import config
from lab_4.src.bot.handlers import handlers
bot = Bot(token="7903937342:AAErWCbbve88OaGhIlRfU3dcBoTOyBTHWiw")

async def main():
    dp = Dispatcher()
    dp.include_router(handlers.rt)
    print("polling started")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())