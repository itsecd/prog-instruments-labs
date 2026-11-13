from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Callable, Iterable

import config_loader as settings


@dataclass(frozen=True)
class AvalancheQuality:
    level: str
    title: str
    description: str


@dataclass(frozen=True)
class AvalancheResult:
    original_text: str
    modified_text: str
    original_hash: str
    modified_hash: str
    changed_bits: int
    total_bits: int
    diff_percent: float
    modification_type: str


def compute_hash(text: str, algorithm: str = settings.DEFAULT_ALGORITHM) -> str:
    """Посчитать хеш строки выбранным алгоритмом."""
    if algorithm not in settings.SUPPORTED_ALGORITHMS:
        raise ValueError(f"Неподдерживаемый алгоритм: {algorithm}")
    h = hashlib.new(algorithm)
    h.update(text.encode(settings.TEXT_ENCODING))
    return h.hexdigest()


def hash_to_bits(hash_hex: str) -> str:
    """Преобразовать hex-представление хеша в строку битов."""
    return "".join(f"{int(char, 16):0{settings.HEX_DIGIT_BITS}b}" for char in hash_hex)


def count_differing_bits(hash_a: str, hash_b: str) -> int:
    """Подсчитать количество различающихся битов в двух hex-хешах."""
    bits_a = hash_to_bits(hash_a)
    bits_b = hash_to_bits(hash_b)
    if len(bits_a) != len(bits_b):
        raise ValueError("Длины хешей не совпадают")
    return sum(a != b for a, b in zip(bits_a, bits_b))


def diff_percent(changed_bits: int, total_bits: int) -> float:
    """Вернуть процент изменившихся битов."""
    if total_bits == 0:
        return 0.0
    return changed_bits * 100 / total_bits


def change_one_char(text: str, position: int | None = None) -> tuple[str, int]:
    """Заменить один символ строки на другой печатный символ."""
    if not text:
        raise ValueError("Нельзя изменить символ в пустой строке")

    pos = random.randrange(len(text)) if position is None else position % len(text)
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    current = text[pos]
    replacement = random.choice(alphabet)
    while replacement == current:
        replacement = random.choice(alphabet)

    return text[:pos] + replacement + text[pos + 1 :], pos


def change_one_bit(text: str) -> tuple[str, int, int]:
    """Инвертировать один случайный бит в UTF-8-представлении строки."""
    if not text:
        raise ValueError("Нельзя изменить бит в пустой строке")

    data = bytearray(text.encode(settings.TEXT_ENCODING))
    byte_i = random.randrange(len(data))
    bit_i = random.randrange(settings.BYTE_BITS)
    data[byte_i] ^= 1 << bit_i
    modified = data.decode(settings.TEXT_ENCODING, errors=settings.DECODE_ERRORS)
    return modified, byte_i, bit_i


def change_case(text: str, position: int | None = None) -> tuple[str, int]:
    """Изменить регистр одной буквы."""
    letter_positions = [i for i, char in enumerate(text) if char.isalpha()]
    if not letter_positions:
        raise ValueError("В строке нет букв для смены регистра")

    if position is None:
        pos = random.choice(letter_positions)
    else:
        pos = letter_positions[position % len(letter_positions)]

    char = text[pos]
    replacement = char.lower() if char.isupper() else char.upper()
    return text[:pos] + replacement + text[pos + 1 :], pos


def _apply_modification(text: str, modification_type: str) -> tuple[str, str]:
    match modification_type:
        case "char":
            modified, pos = change_one_char(text)
            label = settings.MODIFICATION_LABELS["char"].format(pos=pos)
        case "bit":
            modified, byte_i, bit_i = change_one_bit(text)
            label = settings.MODIFICATION_LABELS["bit"].format(byte_i=byte_i, bit_i=bit_i)
        case "case":
            modified, pos = change_case(text)
            label = settings.MODIFICATION_LABELS["case"].format(pos=pos)
        case _:
            raise ValueError(f"Неизвестный тип модификации: {modification_type}")
    return modified, label


def run_single_experiment(
    text: str,
    modification_type: str,
    algorithm: str = settings.DEFAULT_ALGORITHM,
) -> AvalancheResult:
    """Выполнить один эксперимент для выбранного типа изменения."""
    modified_text, label = _apply_modification(text, modification_type)
    original_hash = compute_hash(text, algorithm)
    modified_hash = compute_hash(modified_text, algorithm)
    changed_bits = count_differing_bits(original_hash, modified_hash)
    total_bits = len(hash_to_bits(original_hash))

    return AvalancheResult(
        original_text=text,
        modified_text=modified_text,
        original_hash=original_hash,
        modified_hash=modified_hash,
        changed_bits=changed_bits,
        total_bits=total_bits,
        diff_percent=diff_percent(changed_bits, total_bits),
        modification_type=label,
    )


def run_experiments(
    text: str,
    count: int = settings.DEFAULT_EXPERIMENT_COUNT,
    algorithm: str = settings.DEFAULT_ALGORITHM,
    modifications: Iterable[str] = settings.MODIFICATION_TYPES,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[AvalancheResult]:
    """Запустить серию экспериментов."""
    if count < settings.MIN_EXPERIMENT_COUNT:
        raise ValueError(f"Количество экспериментов должно быть >= {settings.MIN_EXPERIMENT_COUNT}")

    modification_list = tuple(modifications)
    total = count * len(modification_list)
    results: list[AvalancheResult] = []

    for mod in modification_list:
        for _ in range(count):
            results.append(run_single_experiment(text, mod, algorithm))
            if progress_callback is not None:
                progress_callback(len(results), total)

    return results


def summarize_results(results: list[AvalancheResult]) -> dict:
    """Сформировать сводную статистику по экспериментам."""
    if not results:
        return {}

    total_experiments = len(results)
    avg_diff = sum(r.diff_percent for r in results) / total_experiments
    avg_bits = round(sum(r.changed_bits for r in results) / total_experiments, 2)
    total_bits = results[0].total_bits

    by_modification: dict[str, float] = {}
    for mod in dict.fromkeys(r.modification_type for r in results):
        mod_results = [r for r in results if r.modification_type == mod]
        by_modification[mod] = round(
            sum(r.diff_percent for r in mod_results) / len(mod_results),
            2,
        )

    return {
        "total_experiments": total_experiments,
        "avg_diff_percent": round(avg_diff, 2),
        "min_diff_percent": round(min(r.diff_percent for r in results), 2),
        "max_diff_percent": round(max(r.diff_percent for r in results), 2),
        "avg_changed_bits": avg_bits,
        "total_bits": total_bits,
        "by_modification": by_modification,
    }


def get_avalanche_quality(percent: float) -> AvalancheQuality:
    """Оценить качество лавинного эффекта по проценту изменившихся битов."""
    if percent >= settings.EXCELLENT_AVALANCHE_PERCENT:
        level = "excellent"
    elif percent >= settings.MODERATE_AVALANCHE_PERCENT:
        level = "moderate"
    else:
        level = "weak"

    data = settings.QUALITY_LEVELS[level]
    return AvalancheQuality(level=level, title=data["title"], description=data["description"])

