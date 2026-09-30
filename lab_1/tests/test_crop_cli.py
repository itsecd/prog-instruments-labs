import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from crop_cli import parse_arguments, validate_arguments


def test_arguments_are_parsed_and_validated(tmp_path: Path) -> None:
    source = tmp_path / "image.png"
    Image.new("RGB", (5, 4)).save(source)
    args = parse_arguments(
        ["--input", str(source), "--output", "result.png", "--width", "3", "--height", "2"]
    )
    assert (args.width, args.height) == (3, 2)
    assert validate_arguments(args) is None


@pytest.mark.parametrize("scenario", ["success", "missing", "invalid-size", "corrupt"])
def test_cli_exit_status_and_output(tmp_path: Path, scenario: str) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    if scenario == "corrupt":
        source.write_text("invalid image", encoding="utf-8")
    elif scenario != "missing":
        Image.new("RGB", (5, 4), "red").save(source)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crop_cli",
            "--input",
            str(source),
            "--output",
            str(output),
            "--width",
            "0" if scenario == "invalid-size" else "3",
            "--height",
            "2",
        ],
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == (0 if scenario == "success" else 1), result.stderr
    assert output.exists() == (scenario == "success")
    if scenario == "success":
        with Image.open(output) as saved:
            assert saved.size == (3, 2)


def test_cli_requires_arguments() -> None:
    with pytest.raises(SystemExit) as error:
        parse_arguments([])
    assert error.value.code == 2
