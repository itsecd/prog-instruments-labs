import asyncio
import aiofiles
import os


async def read_file(file_name):
    """Читает содержимое файла."""
    print(f"Чтение файла {file_name}...")
    await asyncio.sleep(0.1)  # Небольшая задержка для демонстрации асинхронности
    async with aiofiles.open(file_name, mode='r', encoding='utf-8') as f:
        content = await f.read()
    print(f"Файл {file_name} прочитан!")
    return file_name, content