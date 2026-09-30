from pathlib import Path

from PIL import Image


def load_image(image_path: str) -> Image.Image:
    """Загружает изображение из файла."""
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Файл {image_path} не найден")

    try:
        with Image.open(image_path) as img:
            return img.copy()
    except OSError as e:
        raise OSError(f"Ошибка при загрузке изображения: {e}") from e


def get_image_size(img: Image.Image) -> tuple[int, int]:
    """Получает размер изображения."""
    return img.size


def crop_image_from_top_left(img: Image.Image, width: int, height: int) -> Image.Image:
    """Обрезает изображение от левого верхнего угла."""
    if width <= 0 or height <= 0:
        raise ValueError("Ширина и высота должны быть положительными числами")

    original_width, original_height = img.size

    crop_width = min(width, original_width)
    crop_height = min(height, original_height)

    if crop_width != width or crop_height != height:
        print(
            f"Предупреждение: заданные размеры ({width}x{height}) "
            f"превышают размеры изображения ({original_width}x{original_height})"
        )
        print(f"Используются размеры: {crop_width}x{crop_height}")

    cropped_img = img.crop((0, 0, crop_width, crop_height))
    return cropped_img


def save_image(img: Image.Image, output_path: str) -> None:
    """Сохраняет изображение в файл."""
    try:
        # Создаёт каталог для результата.
        output_dir = Path(output_path).parent
        if output_dir != Path("."):
            output_dir.mkdir(parents=True, exist_ok=True)

        img.save(output_path)
        print(f"Результат сохранен в: {output_path}")
    except (OSError, ValueError) as e:
        raise OSError(f"Ошибка при сохранении изображения: {e}") from e


def process_image_crop(
    input_path: str, output_path: str, width: int, height: int
) -> tuple[Image.Image, Image.Image]:
    """Загружает, обрезает и сохраняет изображение."""
    print(f"Загрузка изображения: {input_path}")
    img = load_image(input_path)

    original_width, original_height = get_image_size(img)
    print(f"Размер исходного изображения: {original_width}x{original_height} пикселей")

    cropped_img = crop_image_from_top_left(img, width, height)
    crop_width, crop_height = get_image_size(cropped_img)
    print(f"Размер обрезанного изображения: {crop_width}x{crop_height} пикселей")

    save_image(cropped_img, output_path)

    return img, cropped_img
