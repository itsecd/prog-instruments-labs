from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from PIL import Image

from visualization import display_images_comparison, save_comparison_plot


def test_save_comparison_and_close_figure(tmp_path: Path) -> None:
    output = tmp_path / "comparison.png"
    before = plt.get_fignums()
    save_comparison_plot(Image.new("RGB", (5, 4)), Image.new("RGB", (3, 2)), str(output))
    with Image.open(output) as saved:
        assert saved.format == "PNG"
        assert saved.width > 100
        assert saved.height > 100
    assert plt.get_fignums() == before


def test_failed_save_closes_figure(tmp_path: Path) -> None:
    before = plt.get_fignums()
    with pytest.raises(OSError):
        save_comparison_plot(
            Image.new("RGB", (5, 4)),
            Image.new("RGB", (3, 2)),
            str(tmp_path / "missing" / "comparison.png"),
        )
    assert plt.get_fignums() == before


def test_display_closes_figure(monkeypatch: pytest.MonkeyPatch) -> None:
    before = plt.get_fignums()
    calls: list[bool] = []

    def show() -> None:
        calls.append(True)

    monkeypatch.setattr(plt, "show", show)
    display_images_comparison(Image.new("RGB", (5, 4)), Image.new("RGB", (3, 2)))
    assert calls == [True]
    assert plt.get_fignums() == before
