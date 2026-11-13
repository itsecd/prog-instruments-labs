from unittest.mock import patch

import pytest

from src.ast import NumberTree, OperatorTree
from src.evaluator import Evaluator
from src.exception import CalcException
from src.parser import Parser
from src.scanner import Scanner


def evaluate_expr(expr):
    tokens = Scanner().scan_expr(expr + "\n")
    return Evaluator().eval(Parser().parse(tokens))


def test_number_tree_evaluates_to_its_number():
    assert Evaluator().eval(NumberTree(42)) == 42


@pytest.mark.parametrize(
    "operator, lhs, rhs, expected",
    [
        ("+", 2, 3, 5),
        ("-", 2, 3, -1),
        ("*", 4, 5, 20),
        ("/", 8, 2, 4),
        ("/", 7, 2, 3.5),
    ],
)
def test_operator_tree_evaluation(operator, lhs, rhs, expected):
    tree = OperatorTree(operator, NumberTree(lhs), NumberTree(rhs))
    assert Evaluator().eval(tree) == expected


def test_nested_tree_is_evaluated_recursively():
    tree = OperatorTree(
        "*",
        OperatorTree("+", NumberTree(1), NumberTree(2)),
        NumberTree(4),
    )
    assert Evaluator().eval(tree) == 12


@pytest.mark.parametrize("numerator", [7, 8])
def test_division_by_zero_raises_calc_exception(numerator):
    tree = OperatorTree("/", NumberTree(numerator), NumberTree(0))
    with pytest.raises(CalcException, match="divide by zero"):
        Evaluator().eval(tree)


def test_eval_visits_every_node_of_the_tree():
    tree = OperatorTree("+", NumberTree(1), NumberTree(2))
    original_eval = Evaluator.eval

    with patch.object(
        Evaluator, "eval", autospec=True, side_effect=original_eval
    ) as spy:
        result = Evaluator().eval(tree)

    assert result == 3
    assert spy.call_count == 3  # the operator node and both number leaves


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("2+3*4", 14),
        ("(2+3)*4", 20),
        ("2*(3+4)-5", 9),
        ("100-1-1", 98),
        ("1.5+2.5", 4.0),
    ],
)
def test_full_pipeline_from_text_to_result(expr, expected):
    assert evaluate_expr(expr) == pytest.approx(expected)


@pytest.mark.xfail(strict=True, reason="_div uses floor division for even numerators")
@pytest.mark.parametrize("expr, expected", [("10/4", 2.5), ("4/8", 0.5)])
def test_division_of_even_numerator_should_not_be_truncated(expr, expected):
    assert evaluate_expr(expr) == pytest.approx(expected)