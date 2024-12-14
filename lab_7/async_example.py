import asyncio
import aiofiles

from collections import Counter


async def read_file(file_name):
    """Читает содержимое файла."""
    async with aiofiles.open(file_name, mode='r', encoding='utf-8') as f:
        content = await f.read()
    return file_name, content


async def analyze_file(file_name, content):
    """Анализирует содержимое файла."""

    num_lines = content.count('\n') + 1
    num_words = len(content.split())
    num_chars = len(content)

    words = content.split()
    word_freq = Counter(words)

    lines = content.splitlines()
    longest_lines = sorted(lines, key=len, reverse=True)[:3]
    
    return {
        "file_name": file_name,
        "num_lines": num_lines,
        "num_words": num_words,
        "num_chars": num_chars,
        "word_freq": word_freq,
        "longest_lines": longest_lines,
    }


async def save_analysis(file_name, analysis_data):
    """Сохраняет результаты анализа в файл."""
    output_file = f"{file_name}_analysis.txt"
    async with aiofiles.open(output_file, mode='w', encoding='utf-8') as f:
        await f.write(f"Результаты анализа файла {file_name}:\n")
        await f.write(f"Количество строк: {analysis_data['num_lines']}\n")
        await f.write(f"Количество слов: {analysis_data['num_words']}\n")
        await f.write(f"Количество символов: {analysis_data['num_chars']}\n\n")
        
        await f.write("Самые часто встречающиеся слова:\n")
        for word, count in analysis_data['word_freq'].most_common(5):
            await f.write(f"  {word}: {count}\n")
        
        await f.write("\nСамые длинные строки:\n")
        for line in analysis_data['longest_lines']:
            await f.write(f"  {line}\n")
    
    print(f"Результаты сохранены в файл {output_file}")


async def main():
    # Создаём несколько текстовых файлов для примера
    file_names = ['file_1.txt', 'file_2.txt', 'file_3.txt']
    file_contents = [
        "Hello, this is the first file.\nIt has two lines.\nAnd some repeated words like file and file.",
        "This is the second file.\nIt has three lines.\nHere's the third line.\nFile content is fun!",
        "File number three is here.\nIt has four lines.\nLine three.\nLine four.\nRepetition is key. Key is repetition.",
    ]
    
    for file_name, content in zip(file_names, file_contents):
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(content)

    read_tasks = [read_file(file_name) for file_name in file_names]
    file_data = await asyncio.gather(*read_tasks)

    analyze_tasks = [analyze_file(file_name, content) for file_name, content in file_data]
    analysis_results = await asyncio.gather(*analyze_tasks)

    save_tasks = [save_analysis(result["file_name"], result) for result in analysis_results]
    await asyncio.gather(*save_tasks)
    
    print("\nАнализ завершён! Результаты сохранены.")


asyncio.run(main())