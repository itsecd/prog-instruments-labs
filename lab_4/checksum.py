import json
import hashlib
from typing import List
from pathlib import Path


"""
В этом модуле обитают функции, необходимые для автоматизированной проверки
результатов ваших трудов.
"""


def calculate_checksum(row_numbers: List[int]) -> str:
    """
    Вычисляет md5 хеш от списка целочисленных значений.

    ВНИМАНИЕ, ВАЖНО! Чтобы сумма получилась корректной, считать, что первая
    строка с данными csv-файла имеет номер 0
    Другими словами: В исходном csv 1я строка - заголовки столбцов, 2я и
    остальные - данные.
    Соответственно, считаем что у 2 строки файла номер 0, у 3й - номер 1
    и так далее.

    Надеюсь, я расписал это максимально подробно.
    Хотя что-то мне подсказывает, что обязательно найдется человек, у которого
    с этим возникнут проблемы.
    Которому я отвечу, что все написано в докстринге

    :param row_numbers: список целочисленных номеров строк csv-файла, на
    которых были найдены ошибки валидации
    :return: md5 хеш для проверки через github action
    """
    row_numbers.sort()
    return hashlib.md5(json.dumps(row_numbers).encode('utf-8')).hexdigest()


def serialize_result(variant: int, checksum: str) -> None:
    """
    Записывает номер варианта и контрольную сумму в result.json.

    Читает существующий файл, чтобы не потерять дополнительные ключи
    и сохранить исходную структуру. Если файла нет — создаёт минимальный.

    :param variant: номер варианта
    :param checksum: контрольная сумма из calculate_checksum()
    """
    result_path = Path(__file__).resolve().parent / "result.json"

    if result_path.exists():
        with result_path.open(encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}

    data["variant"] = variant
    data["checksum"] = checksum

    with result_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")  # финальный перевод строки — хорошая практика


if __name__ == "__main__":
    print(calculate_checksum([1, 2, 3]))
    print(calculate_checksum([3, 2, 1]))
