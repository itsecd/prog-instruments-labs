from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from numpy.typing import NDArray

from audio_utils import (
    get_duration,
    load_audio,
    save_audio,
    trim_audio,
    validate_range,
)


def test_get_duration(audio: NDArray[np.float64], samplerate: int) -> None:
    assert get_duration(audio, samplerate) == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        (0.0, 2.0, True),
        (0.5, 1.5, True),
        (1.0, 1.0, False),
        (1.5, 0.5, False),
        (-0.1, 1.0, False),
        (0.0, 2.1, False),
    ],
)
def test_validate_range(start: float, end: float, expected: bool) -> None:
    assert validate_range(start, end, 2.0) is expected


def test_trim_audio_length(audio: NDArray[np.float64], samplerate: int) -> None:
    trimmed = trim_audio(audio, samplerate, 0.5, 1.5)

    assert len(trimmed) == samplerate


def test_trim_audio_values(audio: NDArray[np.float64], samplerate: int) -> None:
    trimmed = trim_audio(audio, samplerate, 0.5, 1.5)

    first = samplerate // 2
    last = first + samplerate
    np.testing.assert_array_equal(trimmed, audio[first:last])


def test_save_and_load_roundtrip(
    tmp_path: Path, audio: NDArray[np.float64], samplerate: int
) -> None:
    target = tmp_path / "nested" / "dir" / "out.wav"

    save_audio(audio, samplerate, str(target))
    loaded, loaded_rate = load_audio(str(target))

    assert loaded_rate == samplerate
    np.testing.assert_allclose(loaded, audio, atol=1e-4)


def test_save_audio_to_current_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    audio: NDArray[np.float64],
    samplerate: int,
) -> None:
    monkeypatch.chdir(tmp_path)

    save_audio(audio, samplerate, "out.wav")

    assert (tmp_path / "out.wav").is_file()


def test_load_audio_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_audio(str(tmp_path / "missing.wav"))


def test_load_audio_not_audio(tmp_path: Path) -> None:
    fake = tmp_path / "fake.wav"
    fake.write_text("это не аудио", encoding="utf-8")

    with pytest.raises(sf.SoundFileError):
        load_audio(str(fake))
