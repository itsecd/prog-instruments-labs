import asyncio
from aiogram import Bot, Dispatcher

#local imports
import config
from handlers import handlers
bot = Bot(token=config.TOKEN)

async def main():
    dp = Dispatcher()
    dp.include_router(handlers.rt)
    print("polling started")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())