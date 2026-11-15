from main import build_url
import pytest
from bs4 import BeautifulSoup
from main import extract_audio_urls
import csv
from main import write_csv_annotation
from main import FileIterator


def test_build_url():
    result = build_url("rain")

    assert result == "https://mixkit.co/free-sound-effects/rain/"


@pytest.mark.parametrize(
    ("html", "expected"),
    [
        ('<div data-audio="sound.mp3"></div>', ["sound.mp3"]),
        ('<div data-audio=""></div>', []),
        ("<div></div>", []),
    ],
)

def test_extract_audio_urls(html, expected):
    soup = BeautifulSoup(html, "html.parser")

    result = extract_audio_urls(soup, "data-audio")

    assert result == expected

def test_write_csv_annotation(tmp_path):
    sound = tmp_path / "rain.mp3"
    sound.touch()
    csv_path = tmp_path / "annotation.csv"

    write_csv_annotation([sound], csv_path, tmp_path)

    with csv_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert rows == [
        {
            "filename": "rain.mp3",
            "absolute_path": str(sound.resolve()),
            "relative_path": "rain.mp3",
        }
    ]

def test_file_iterator_finds_mp3(tmp_path):
    first = tmp_path / "rain.mp3"
    first.touch()

    nested = tmp_path / "nested"
    nested.mkdir()
    second = nested / "wind.mp3"
    second.touch()

    (tmp_path / "notes.txt").touch()

    result = list(FileIterator(tmp_path))

    assert set(result) == {first, second}

def test_file_iterator_reject_source():
    with pytest.raises(TypeError, match="Source must be path"):
        FileIterator("sounds")