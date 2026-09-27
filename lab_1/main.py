"""Модуль запуска основного приложения."""

import argparse

import soundfile as sf

from audio_utils import (
    get_duration,
    load_audio,
    save_audio,
    trim_audio,
    validate_range,
)
from visualizer import plot_audio_comparison


def parse_args() -> argparse.Namespace:
    """Парсит аргументы командной строки."""
    parser = argparse.ArgumentParser(description="Обрезка аудио в заданном диапазоне")
    parser.add_argument("--input", required=True, help="Путь к исходному файлу")
    parser.add_argument("--output", required=True, help="Путь для сохранения файла")
    parser.add_argument(
        "--start", type=float, required=True, help="Начало диапазона(сек)"
    )
    parser.add_argument("--end", type=float, required=True, help="Конец диапазона(сек)")
    return parser.parse_args()


def main() -> None:
    """Main функция."""
    try:
        args = parse_args()
        try:
            data, samplerate = load_audio(args.input)
        except FileNotFoundError:
            print(f"Ошибка: файл '{args.input}' не найден.")
            return
        except sf.SoundFileError as e:
            print(f"Ошибка при загрузке аудио: {e}")
            return
        duration = get_duration(data, samplerate)
        print(f"Длительность исходного аудиофайла: {duration:.2f} сек.")
        if not validate_range(args.start, args.end, duration):
            print("Ошибка: некорректный диапазон.")
            return
        trimmed_audio = trim_audio(data, samplerate, args.start, args.end)
        try:
            save_audio(trimmed_audio, samplerate, args.output)
            print(f"Обрезанное аудио сохранено в: {args.output}")
        except (sf.SoundFileError, TypeError, OSError) as e:
            print(f"Ошибка при сохранении файла: {e}")
            return
        plot_audio_comparison(data, trimmed_audio, samplerate, args.start, args.end)
    except KeyboardInterrupt:
        print("\nПроцесс прерван пользователем.")


if __name__ == "__main__":
    main()
