import asyncio
import aiofiles
import os

from collections import Counter


async def read_file(file_name):
    """Читает содержимое файла."""
    async with aiofiles.open(file_name, mode='r', encoding='utf-8') as f:
        content = await f.read()
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


async def main():
    # Создаём несколько текстовых файлов для примера
    file_names = ['file_1.txt', 'file_2.txt', 'file_3.txt']
    file_contents = [
        "Hello, this is the first file.\nIt has two lines.",
        "This is the second file.\nIt has three lines.\nHere's the third line.",
        "File number three is here.\nIt has four lines.\nLine three.\nLine four.",
    ]
    
    # Создание файлов (можно пропустить, если файлы уже существуют)
    for file_name, content in zip(file_names, file_contents):
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(content)
    
    # Чтение файлов асинхронно
    read_tasks = [read_file(file_name) for file_name in file_names]
    file_data = await asyncio.gather(*read_tasks)
    
    # Анализ содержимого файлов
    analyze_tasks = [analyze_file(file_name, content) for file_name, content in file_data]
    analysis_results = await asyncio.gather(*analyze_tasks)
    
    print("\nРезультаты анализа:")
    for file_name, num_lines, num_words, num_chars in analysis_results:
        print(f"{file_name}:")
        print(f"  Строк: {num_lines}")
        print(f"  Слов: {num_words}")
        print(f"  Символов: {num_chars}")


asyncio.run(main())