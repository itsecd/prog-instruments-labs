import os
from typing import List
from log_filter import (
    filter_logs_time,
    count_game_launches,
    filter_logs_function,
    filter_logs_by_regex
)


def create_output_directory(directory: str) -> None:
    """
    Создает директорию для выходных файлов, если она не существует.

    Parameters:
        directory (str): Путь к директории для выходных файлов.
    """
    os.makedirs(directory, exist_ok=True)


def filter_logs_by_time(input_file: str, output_file: str, start_time: str, end_time: str) -> None:
    """
    Фильтрует логи по времени и сохраняет результат в выходной файл.

    Parameters:
        input_file (str): Путь к входному файлу с логами.
        output_file (str): Путь к выходному файлу для записи отфильтрованных логов.
        start_time (str): Начальное время для фильтрации.
        end_time (str): Конечное время для фильтрации.
    """
    filter_logs_time(start_time, end_time, input_file, output_file)
    print(f"Фильтрация по времени завершена. Отфильтрованные логи сохранены в '{output_file}'.")


def count_game_launches_wrapper(input_file: str, game_name: str) -> int:
    """
    Подсчитывает количество запусков игры.

    Parameters:
        input_file (str): Путь к входному файлу с логами.
        game_name (str): Название игры для подсчета.

    Returns:
        int: Количество запусков игры.
    """
    return count_game_launches(game_name, input_file)


def filter_logs_by_function(input_file: str, output_file: str, function_name: str) -> None:
    """
    Фильтрует логи по указанной функции и сохраняет результат в выходной файл.

    Parameters:
        input_file (str): Путь к входному файлу с логами.
        output_file (str): Путь к выходному файлу для записи отфильтрованных логов.
        function_name (str): Название функции для фильтрации.
    """
    filter_logs_function(function_name, input_file, output_file)
    print(f"Фильтрация по функции '{function_name}' завершена. Отфильтрованные логи сохранены в '{output_file}'.")


def filter_logs_by_regex_patterns(input_file: str, patterns: List[str], output_directory: str) -> None:
    """
    Фильтрует логи по списку регулярных выражений и сохраняет результаты в выходные файлы.

    Parameters:
        input_file (str): Путь к входному файлу с логами.
        patterns (List[str]): Список регулярных выражений для фильтрации.
        output_directory (str): Директория для сохранения выходных файлов.
    """
    for i, pattern in enumerate(patterns, start=1):
        output_file = os.path.join(output_directory, f'filtered_logs_{i}.txt')
        filter_logs_by_regex(input_file, output_file, pattern)
        print(f"Фильтрация по регулярному выражению '{pattern}' завершена. Отфильтрованные логи сохранены в '{output_file}'.")


def main() -> None:
    input_file = 'logs.txt'                   # Имя входного файла с логами
    output_directory = 'output'               # Директория для выходных файлов

    create_output_directory(output_directory)

    print("-----------------")

    # Фильтрация по времени
    start_time = '2024-12-07 19:09:20,000'  # Начальное время
    end_time = '2024-12-07 19:09:40,000'    # Конечное время
    output_file1 = os.path.join(output_directory, 'filtered_logs1.txt')
    filter_logs_by_time(input_file, output_file1, start_time, end_time)

    print("-----------------")

    # Подсчет запусков игры
    game_name = "2048"  # Название игры, которую нужно проверить
    launch_count = count_game_launches_wrapper(input_file, game_name)
    print(f"Игра '{game_name}' была запущена {launch_count} раз(а).")

    print("-----------------")

    # Фильтрация по функции
    output_file2 = os.path.join(output_directory, 'filtered_logs2.txt')
    function_to_track = 'slide_and_merge'
    filter_logs_by_function(input_file, output_file2, function_to_track)

    print("-----------------")

    # Фильтрация по регулярным выражениям
    regex_patterns: List[str] = [
        r'игра \'Виселица\'',
        r'Неверная буква:',
        r'Пользователь угадал букву:',
        r'Игрок проиграл\. Загаданное слово:',
        r'Игра 2048 начата\.',
        r'Пользователь ввел неверное направление:',
        r'Добавлена новая плитка \d+ на позицию',
        r'Доска после движения [wasd]:',
        r'Пользователь .*'
    ]
    filter_logs_by_regex_patterns(input_file, regex_patterns, output_directory)

    print("-----------------")


if __name__ == "__main__":
    main()
