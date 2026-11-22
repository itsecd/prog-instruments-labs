"""Модуль с логикой парсинга логов."""

import csv
import json
import logging
from collections import Counter
from typing import Dict, Tuple

from patterns import LOG_PATTERN

CSV_HEADERS = ['ip', 'identd', 'userid', 'date', 'method', 'url', 'status', 'size']

logger = logging.getLogger("log_parser")


def _parse_line(line: str) -> Dict[str, str] | None:
    """Парсит одну строку лога. Возвращает словарь или None, если строка битая."""
    match = LOG_PATTERN.match(line)
    return match.groupdict() if match else None


def _write_valid_row(writer: csv.writer, data: Dict[str, str]) -> None:
    """Записывает валидную строку в CSV."""
    writer.writerow([data[field] for field in CSV_HEADERS])


def _write_broken_row(f_broken, line_num: int, line: str) -> None:
    """Записывает битую строку в файл."""
    f_broken.write(f"Строка {line_num}: {line}\n")
    logger.warning("Битая строка %d: %s", line_num, line)


def _update_stats(status_counter: Counter, data: Dict[str, str]) -> None:
    """Обновляет счетчик статус-кодов."""
    status_counter[data['status']] += 1


def _save_stats(stats: Dict, stats_json: str) -> None:
    """Сохраняет статистику в JSON-файл."""
    with open(stats_json, 'w', encoding='utf-8') as f_stats:
        json.dump(stats, f_stats, indent=4, ensure_ascii=False)
    logger.info("Статистика сохранена в %s", stats_json)


def _process_lines(
    input_file: str,
    f_valid,
    f_broken,
    status_counter: Counter
) -> Tuple[int, int]:
    """Обрабатывает все строки файла. Возвращает (valid_count, broken_count)."""
    writer = csv.writer(f_valid)
    writer.writerow(CSV_HEADERS)

    valid_count = 0
    broken_count = 0

    with open(input_file, 'r', encoding='utf-8') as f_in:
        for line_num, line in enumerate(f_in, start=1):
            line = line.strip()
            if not line:
                continue

            data = _parse_line(line)
            if data:
                _write_valid_row(writer, data)
                _update_stats(status_counter, data)
                valid_count += 1
            else:
                _write_broken_row(f_broken, line_num, line)
                broken_count += 1

    return valid_count, broken_count


def parse_logs(input_file: str, valid_csv: str, broken_txt: str, stats_json: str) -> Dict:
    """
    Парсит лог-файл, разделяет на валидные и битые, собирает статистику.
    """
    logger.info("Начинаю парсинг файла: %s", input_file)

    status_counter = Counter()

    try:
        with open(valid_csv, 'w', newline='', encoding='utf-8') as f_valid, \
             open(broken_txt, 'w', encoding='utf-8') as f_broken:
            valid_count, broken_count = _process_lines(
                input_file, f_valid, f_broken, status_counter
            )
    except FileNotFoundError:
        logger.error("Файл не найден: %s", input_file)
        raise
    except PermissionError:
        logger.error("Нет прав на чтение файла: %s", input_file)
        raise
    except Exception as e:
        logger.exception("Непредвиденная ошибка при парсинге: %s", e)
        raise

    stats = {
        "total_lines_processed": valid_count + broken_count,
        "valid_lines": valid_count,
        "broken_lines": broken_count,
        "status_codes": dict(status_counter)
    }

    logger.info(
        "Парсинг завершён. Валидных строк: %d, битых: %d",
        valid_count, broken_count
    )
    logger.info("Статистика по статус-кодам: %s", dict(status_counter))

    _save_stats(stats, stats_json)
    return stats