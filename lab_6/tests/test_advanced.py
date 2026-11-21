import argparse

import pytest

import main


@pytest.mark.parametrize(
    "data, expected",
    [
        ("", []),
        ("Фамилия: Иванов\nВозраст: 20", []),
        (
            "Фамилия: петров\nИмя: Анна\n2)\nФамилия: Иванов\nИмя: Борис",
            ["Фамилия: Иванов\nИмя: Борис", "Фамилия: петров\nИмя: Анна"],
        ),
    ],
)
def test_parse_data_parametrized(data, expected):
    assert main.parse_data(data) == expected


def test_main_with_monkeypatch(monkeypatch, capsys):
    monkeypatch.setattr(
        main,
        "args_parse",
        lambda: argparse.Namespace(file="input.txt", output="output.txt"),
    )

    def fake_read_file(filename):
        assert filename == "input.txt"
        return "Фамилия: Петров\nИмя: Иван"

    monkeypatch.setattr(main, "read_file", fake_read_file)

    saved = {}

    def fake_write_file(filename, data):
        saved["filename"] = filename
        saved["data"] = data

    monkeypatch.setattr(main, "write_file", fake_write_file)

    main.main()

    assert saved["filename"] == "output.txt"
    assert saved["data"] == ["Фамилия: Петров\nИмя: Иван"]
    assert capsys.readouterr().out == "Sorted data written to output.txt\n"
