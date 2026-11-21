
from main import read_file, write_file


def test_read_file_returns_content(tmp_path):
    file = tmp_path / "input.txt"
    file.write_text("Привет, мир!", encoding="utf-8")

    result = read_file(str(file))

    assert result == "Привет, мир!"


def test_read_file_missing_file(tmp_path, capsys):
    file = tmp_path / "missing.txt"

    result = read_file(str(file))

    assert result is None
    assert capsys.readouterr().out == "File not found\n"


def test_write_file_numbers_records(tmp_path):
    file = tmp_path / "output.txt"

    write_file(str(file), ["Первая анкета", "Вторая анкета"])

    assert file.read_text(encoding="utf-8") == (
        "1)\nПервая анкета\n\n"
        "2)\nВторая анкета\n\n"
    )
