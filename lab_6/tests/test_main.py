import argparse
import sys

import pytest

import main


def test_main_without_input_file(monkeypatch, capsys):
    monkeypatch.setattr(
        main, "args_parse", lambda: argparse.Namespace(file=None, output=None)
    )

    main.main()

    assert capsys.readouterr().out == "Please specify input file with -f option\n"


def test_main_with_empty_file(monkeypatch, capsys):
    monkeypatch.setattr(
        main,
        "args_parse",
        lambda: argparse.Namespace(file="input.txt", output=None),
    )
    monkeypatch.setattr(main, "read_file", lambda filename: "")

    main.main()

    assert capsys.readouterr().out == ""


def test_main_prints_sorted_forms(monkeypatch, capsys):
    monkeypatch.setattr(
        main,
        "args_parse",
        lambda: argparse.Namespace(file="input.txt", output=None),
    )
    monkeypatch.setattr(
        main,
        "read_file",
        lambda filename: (
            "Фамилия: Петров\nИмя: Олег\n"
            "2)\nФамилия: Иванов\nИмя: Иван"
        ),
    )

    main.main()

    assert capsys.readouterr().out == (
        "1)\nФамилия: Иванов\nИмя: Иван\n\n"
        "2)\nФамилия: Петров\nИмя: Олег\n\n"
    )


@pytest.mark.parametrize(
    "options",
    [
        ["-f", "input.txt", "-o", "output.txt"],
        ["--file", "input.txt", "--output", "output.txt"],
    ],
)
def test_args_parse(monkeypatch, options):
    monkeypatch.setattr(sys, "argv", ["main.py", *options])

    args = main.args_parse()

    assert args.file == "input.txt"
    assert args.output == "output.txt"
