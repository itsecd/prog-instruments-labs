"""Тест."""

import subprocess
import sys
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent.parent
SCRIPT = LAB_DIR / "script.py"
INPUT_FILE = LAB_DIR / "data.txt"
EXPECTED_FILE = Path(__file__).resolve().parent / "expected_output.txt"


def test_output_matches_expected(tmp_path: Path) -> None:
    output_file = tmp_path / "output.txt"

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(INPUT_FILE), str(output_file)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert output_file.read_text(encoding="utf-8") == EXPECTED_FILE.read_text(
        encoding="utf-8"
    )