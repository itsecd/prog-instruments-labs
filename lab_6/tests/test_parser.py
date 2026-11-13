import pytest

from src.ast import NumberTree, OperatorTree
from src.exception import CalcException
from src.parser import Parser
from src.scanner import Scanner


def parse(expr):
    tokens = Scanner().scan_expr(expr + "\n")
    return Parser().parse(tokens)


def to_tuple(node):
    """Convert an AST into nested tuples, e.g. ('+', 1, ('*', 2, 3))."""
    if isinstance(node, NumberTree):
        return node.number
    return (node.operator, to_tuple(node.lhs), to_tuple(node.rhs))


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("7", 7),
        ("((7))", 7),
        ("1+2", ("+", 1, 2)),
        ("1+2*3", ("+", 1, ("*", 2, 3))),
        ("1*2+3", ("+", ("*", 1, 2), 3)),
        ("(1+2)*3", ("*", ("+", 1, 2), 3)),
        ("1-2-3", ("-", ("-", 1, 2), 3)),
        ("8/4/2", ("/", ("/", 8, 4), 2)),
        ("2*(3+4)-5", ("-", ("*", 2, ("+", 3, 4)), 5)),
    ],
)
def test_parse_builds_tree_with_correct_precedence(expr, expected):
    assert to_tuple(parse(expr)) == expected


def test_parse_returns_operator_tree_for_binary_expression():
    assert isinstance(parse("1+2"), OperatorTree)


@pytest.mark.parametrize(
    "expr, message",
    [
        ("1+", "Reached end of tokens"),
        ("(1+2", "missing right parenthesis"),
        ("1+2)", "Unconsumed token"),
        ("3 3", "Unconsumed token"),
        ("()", "Expected a number"),
        ("-5", "Expected a number"),
        ("*2", "Expected a number"),
    ],
)
def test_parse_reports_malformed_expressions(expr, message):
    with pytest.raises(CalcException, match=message):
        parse(expr)


def test_parse_rejects_empty_token_list():
    with pytest.raises(CalcException, match="empty tokens list"):
        Parser().parse([])


def test_parser_consumes_all_tokens():
    parser = Parser()
    parser.parse(Scanner().scan_expr("1+2\n"))
    assert parser.current_token is None