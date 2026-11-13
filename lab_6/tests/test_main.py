import pytest

from main import Calc
from src.ast import NumberTree, OperatorTree


def run_calc(monkeypatch, *lines):
    """Run the REPL feeding it the given input lines."""
    answers = iter(lines)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    return Calc()


def test_exit_command_stops_repl_after_welcome_message(monkeypatch, capsys):
    run_calc(monkeypatch, "exit")

    assert capsys.readouterr().out == "Welcome to calc, type 'exit' to exit calc.\n"


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("1+2", "> 3"),
        ("2+3*4", "> 14"),
        ("(1+2)*3", "> 9"),
        ("1.5+1", "> 2.5"),
        ("  7 - 2 ", "> 5"),
    ],
)
def test_repl_prints_result_of_expression(monkeypatch, capsys, expr, expected):
    run_calc(monkeypatch, expr, "exit")

    assert expected in capsys.readouterr().out.splitlines()


def test_repl_handles_several_expressions_in_one_session(monkeypatch, capsys):
    run_calc(monkeypatch, "1+1", "2*3", "exit")

    lines = capsys.readouterr().out.splitlines()
    assert lines[1:] == ["> 2", "> 6"]


@pytest.mark.parametrize(
    "expr, message",
    [
        ("1/0", "Trying to divide by zero"),
        ("(1+2", "missing right parenthesis"),
        ("2 +", "Reached end of tokens"),
        ("1..2", "multiple decimal points"),
    ],
)
def test_repl_prints_error_and_exits_with_code_1(monkeypatch, capsys, expr, message):
    with pytest.raises(SystemExit) as exit_info:
        run_calc(monkeypatch, expr)

    assert exit_info.value.code == 1
    assert message in capsys.readouterr().out


def test_print_ast_prints_tree_in_prefix_form(monkeypatch, capsys):
    calc = run_calc(monkeypatch, "exit")
    capsys.readouterr()  # drop the welcome message
    tree = OperatorTree("+", NumberTree(1), NumberTree(2))

    calc.print_ast(tree)

    assert capsys.readouterr().out == "(+ 1 2)\n"