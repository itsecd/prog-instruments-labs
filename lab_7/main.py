import re
import argparse
from typing import Iterator, Optional, Tuple

SEPARATOR_RE = re.compile(r'\n\d+\)\n')
BIRTH_YEAR_RE = re.compile(r'Дата рождения:\s*[^\n]*?\d{1,2}[./-]\d{1,2}[./-](\d{2,4})')


def parse_arguments() -> argparse.Namespace:
    """
    Парсит аргументы командной строки.
    """
    parser = argparse.ArgumentParser(description='Анализ анкет людей')
    parser.add_argument('filename', type=str, help='Имя файла с данными')
    return parser.parse_args()


def iter_profiles(filename: str, chunk_size: int = 256 * 1024) -> Iterator[str]:
    """
    Потоково читает файл блоками и по одной возвращает анкеты без загрузки
    всего файла в оперативную память (сложность по памяти O(1)).
    """
    with open(filename, 'r', encoding='utf-8') as file:
        buf = ''
        started = False

        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            buf += chunk
            # Оставляем небольшой хвост, чтобы не разорвать разделитель '\n<число>)\n' на границе блоков
            safe_end = len(buf) - 32
            if safe_end <= 0:
                continue

            pos = 0
            for match in SEPARATOR_RE.finditer(buf, 0, safe_end):
                if started:
                    yield buf[pos:match.start()]
                else:
                    started = True
                pos = match.end()

            if pos > 0:
                buf = buf[pos:]

        pos = 0
        for match in SEPARATOR_RE.finditer(buf):
            if started:
                yield buf[pos:match.start()]
            else:
                started = True
            pos = match.end()

        if started:
            yield buf[pos:]


def is_21st_century_profile(profile_text: str) -> bool:
    """
    За один поиск по предкомпилированному регулярному выражению извлекает год рождения
    и проверяет, родился ли человек в XXI веке (после 2000 года).
    """
    match = BIRTH_YEAR_RE.search(profile_text)
    if match:
        year_str = match.group(1)
        year = int(year_str)
        if len(year_str) == 2:
            year += 2000 if year <= 23 else 1900
        return year > 2000
    return False


def process_and_save_profiles(
    input_filename: str,
    output_filename: str = '21st_century_profiles.txt',
) -> Tuple[Optional[int], Optional[str]]:
    """
    Потоково фильтрует анкеты и сразу сохраняет подходящие в выходной файл.
    """
    count = 0
    out_file = None

    try:
        for profile_text in iter_profiles(input_filename):
            if is_21st_century_profile(profile_text):
                if out_file is None:
                    out_file = open(output_filename, 'w', encoding='utf-8', buffering=256 * 1024)
                count += 1
                out_file.write(f"{count})\n{profile_text}\n")
    except FileNotFoundError:
        print(f"Ошибка: Файл {input_filename} не найден")
        return None, None
    except Exception as e:
        print(f"Ошибка при обработке файла: {e}")
        return None, None
    finally:
        if out_file is not None:
            out_file.close()

    return count, (output_filename if count > 0 else None)


def print_statistics(count: int) -> None:
    """
    Выводит статистику по обработанным анкетам.
    """
    print(f"Количество людей, родившихся в 21 веке: {count}")


def main() -> None:
    """
    Основная функция программы.
    """
    args = parse_arguments()
    count, output_file = process_and_save_profiles(args.filename)
    if count is None:
        return

    print_statistics(count)
    if output_file:
        print(f"Анкеты сохранены в файл: {output_file}")


if __name__ == "__main__":
    main()