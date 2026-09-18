import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from PIL import Image


def _comparison_figure(
    original_img: Image.Image,
    processed_img: Image.Image,
    original_title: str,
    processed_title: str,
) -> Figure:
    """Создаёт график сравнения изображений."""
    fig = plt.figure(figsize=(12, 6))  # pyright: ignore[reportUnknownMemberType]
    original_axes: Axes = fig.add_subplot(1, 2, 1)
    processed_axes: Axes = fig.add_subplot(1, 2, 2)
    for axes, img, title in (
        (original_axes, original_img, original_title),
        (processed_axes, processed_img, processed_title),
    ):
        width, height = img.size
        axes.imshow(img)  # pyright: ignore[reportUnknownMemberType]
        axes.set_title(  # pyright: ignore[reportUnknownMemberType]
            f"{title}\n{width}x{height} px", fontsize=12, fontweight="bold"
        )
        axes.axis("off")
    fig.tight_layout()
    return fig


def display_images_comparison(
    original_img: Image.Image,
    processed_img: Image.Image,
    original_title: str = "Исходное изображение",
    processed_title: str = "Обработанное изображение",
) -> None:
    """Отображает исходное и обработанное изображения рядом."""
    fig = _comparison_figure(original_img, processed_img, original_title, processed_title)
    try:
        plt.show()  # pyright: ignore[reportUnknownMemberType]
    finally:
        plt.close(fig)


def save_comparison_plot(
    original_img: Image.Image,
    processed_img: Image.Image,
    output_path: str,
    original_title: str = "Исходное изображение",
    processed_title: str = "Обработанное изображение",
) -> None:
    """Сохраняет сравнение изображений в файл."""
    fig = _comparison_figure(original_img, processed_img, original_title, processed_title)
    try:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")  # pyright: ignore[reportUnknownMemberType]
    finally:
        plt.close(fig)
    print(f"График сохранен в: {output_path}")
