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


async def analyze_file(file_name, content):
    """Анализирует содержимое файла."""
    print(f"Анализ файла {file_name}...")
    await asyncio.sleep(0.1)  # Имитация обработки для демонстрации
    num_lines = content.count('\n') + 1
    num_words = len(content.split())
    num_chars = len(content)
    print(f"Анализ завершён: {file_name}")
    return file_name, num_lines, num_words, num_chars