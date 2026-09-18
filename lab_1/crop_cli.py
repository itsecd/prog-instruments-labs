import argparse
from collections.abc import Sequence
from pathlib import Path

from image_processing import process_image_crop
from visualization import display_images_comparison


class CropArguments(argparse.Namespace):
    """Хранит аргументы командной строки."""

    input: str
    output: str
    width: int
    height: int


def parse_arguments(argv: Sequence[str] | None = None) -> CropArguments:
    """Разбирает аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Обрезка изображения до заданных размеров от левого верхнего угла",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  %(prog)s --input photo.jpg --output cropped.jpg --width 500 --height 400
  %(prog)s --input ./images/cat.png --output ./output/cat_crop.png --width 300 --height 300
        """,
    )

    parser.add_argument("--input", type=str, required=True, help="Путь к исходному изображению")
    parser.add_argument(
        "--output", type=str, required=True, help="Путь для сохранения обрезанного изображения"
    )
    parser.add_argument(
        "--width", type=int, required=True, help="Ширина обрезанного изображения в пикселях"
    )
    parser.add_argument(
        "--height", type=int, required=True, help="Высота обрезанного изображения в пикселях"
    )

    return parser.parse_args(argv, namespace=CropArguments())


def validate_arguments(args: CropArguments) -> str | None:
    """Проверяет корректность аргументов."""
    if not Path(args.input).exists():
        return f"Ошибка: файл {args.input} не найден"

    if args.width <= 0 or args.height <= 0:
        return "Ошибка: ширина и высота должны быть положительными числами"

    return None


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_arguments(argv)

    # Проверяет аргументы.
    error_message = validate_arguments(args)
    if error_message:
        print(error_message)
        return 1

    # Обрабатывает изображение.
    try:
        original_img, cropped_img = process_image_crop(
            args.input, args.output, args.width, args.height
        )

        # Показывает результат.
        display_images_comparison(
            original_img, cropped_img, "Исходное изображение", "Обрезанное изображение"
        )

        print("", "Обработка завершена успешно!", sep="\n")

    except (OSError, ValueError) as e:
        print(f"Ошибка при обработке изображения: {e}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
