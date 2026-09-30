from pathlib import Path

import pytest
from PIL import Image

from image_processing import (
    crop_image_from_top_left,
    get_image_size,
    load_image,
    process_image_crop,
    save_image,
)


def test_crop_preserves_top_left_pixels() -> None:
    original = Image.new("RGB", (6, 5), "white")
    original.putpixel((0, 0), (255, 0, 0))
    original.putpixel((2, 1), (0, 0, 255))
    cropped = crop_image_from_top_left(original, 3, 2)
    assert get_image_size(cropped) == (3, 2)
    assert cropped.getpixel((0, 0)) == (255, 0, 0)
    assert cropped.getpixel((2, 1)) == (0, 0, 255)
    assert original.size == (6, 5)


@pytest.mark.parametrize("width,height", [(0, 2), (-1, 2), (2, 0), (2, -1)])
def test_crop_rejects_nonpositive_dimensions(width: int, height: int) -> None:
    with pytest.raises(ValueError, match="положительными"):
        crop_image_from_top_left(Image.new("RGB", (5, 4)), width, height)


@pytest.mark.parametrize(
    "width,height,expected", [(20, 2, (5, 2)), (2, 20, (2, 4)), (20, 20, (5, 4))]
)
def test_crop_clamps_to_source(
    width: int, height: int, expected: tuple[int, int], capsys: pytest.CaptureFixture[str]
) -> None:
    result = crop_image_from_top_left(Image.new("RGB", (5, 4)), width, height)
    assert result.size == expected
    assert "Предупреждение" in capsys.readouterr().out


def test_missing_image(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_image(str(tmp_path / "missing.png"))


def test_corrupt_image_preserves_cause(tmp_path: Path) -> None:
    source = tmp_path / "invalid.png"
    source.write_text("not an image", encoding="utf-8")
    with pytest.raises(OSError, match="Ошибка при загрузке") as error:
        load_image(str(source))
    assert error.value.__cause__ is not None


def test_loaded_image_is_independent_of_file(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (5, 4), "red").save(source)
    image = load_image(str(source))
    source.unlink()
    assert image.getpixel((0, 0)) == (255, 0, 0)


def test_process_creates_output_directories(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "nested" / "result.png"
    Image.new("RGB", (5, 4), "red").save(source)
    original, cropped = process_image_crop(str(source), str(output), 3, 2)
    assert original.size == (5, 4)
    assert cropped.size == (3, 2)
    with Image.open(output) as saved:
        assert saved.size == (3, 2)
        assert saved.getpixel((2, 1)) == (255, 0, 0)


def test_unknown_output_format_preserves_cause(tmp_path: Path) -> None:
    with pytest.raises(OSError, match="Ошибка при сохранении") as error:
        save_image(Image.new("RGB", (2, 2)), str(tmp_path / "out.unknown"))
    assert isinstance(error.value.__cause__, ValueError)
