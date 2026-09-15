import subprocess
import sys
from pathlib import Path


def test_script_runs_and_creates_output(tmp_path: Path) -> None:
    """Проверяем, что скрипт запускается и создаёт непустой файл."""
    lab_dir = Path(__file__).parent.parent
    input_file = lab_dir / "data.txt"
    output_file = tmp_path / "result.txt"

    result = subprocess.run(
        [
            sys.executable,
            str(lab_dir / "lab1_var13.py"),
            str(input_file),
            str(output_file),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"Скрипт упал: {result.stderr}"
    assert output_file.exists(), "Выходной файл не создан"
    assert output_file.stat().st_size > 0, "Выходной файл пуст"


def test_output_contains_profiles(tmp_path: Path) -> None:
    """Проверяем, что в результат попали анкеты с телефонами."""
    lab_dir = Path(__file__).parent.parent
    input_file = lab_dir / "data.txt"
    output_file = tmp_path / "result.txt"

    subprocess.run(
        [
            sys.executable,
            str(lab_dir / "lab1_var13.py"),
            str(input_file),
            str(output_file),
        ],
        check=True,
    )

    content = output_file.read_text(encoding="utf-8")
    assert "Фамилия:" in content, "В выводе нет анкет"
