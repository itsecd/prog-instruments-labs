import pytest

from src.exception import CalcException
from src.scanner import Scanner
from src.token import NumberToken, TokenType


def scan(expr):
    """Scan an expression the same way the REPL does (with a trailing newline)."""
    return Scanner().scan_expr(expr + "\n")


def describe(tokens):
    return [(token.type, token.value) for token in tokens]


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("1+2", [(TokenType.NUMBER, 1), (TokenType.PLUS, "+"), (TokenType.NUMBER, 2)]),
        ("4*5", [(TokenType.NUMBER, 4), (TokenType.TIMES, "*"), (TokenType.NUMBER, 5)]),
        ("9/3", [(TokenType.NUMBER, 9), (TokenType.DIV, "/"), (TokenType.NUMBER, 3)]),
        ("7-1", [(TokenType.NUMBER, 7), (TokenType.MINUS, "-"), (TokenType.NUMBER, 1)]),
        (
            "(1)",
            [(TokenType.LPAREN, "("), (TokenType.NUMBER, 1), (TokenType.RPAREN, ")")],
        ),
    ],
)
def test_scan_expr_returns_expected_tokens(expr, expected):
    assert describe(scan(expr)) == expected


@pytest.mark.parametrize("expr", ["1+2", " 1 + 2 ", "1\t+\t2", "  1+  2"])
def test_whitespace_is_ignored(expr):
    assert describe(scan(expr)) == describe(scan("1+2"))


@pytest.mark.parametrize(
    "expr, expected",
    [("42", 42), ("0", 0), ("007", 7), ("2.0", 2), ("3.5", 3.5), (".5", 0.5)],
)
def test_number_values(expr, expected):
    (token,) = scan(expr)
    assert isinstance(token, NumberToken)
    assert token.value == pytest.approx(expected)


def test_integer_valued_float_is_returned_as_int():
    (token,) = scan("2.0")
    assert isinstance(token.value, int)


def test_empty_expression_gives_no_tokens():
    assert scan("") == []


def test_multiple_decimal_points_raise_error():
    with pytest.raises(CalcException, match="multiple decimal points"):
        scan("1..2")


def test_decimal_point_without_digits_raises_error():
    with pytest.raises(CalcException, match="decimal point without digits"):
        scan(".")


@pytest.mark.parametrize(
    "method, char, expected",
    [
        ("is_whitespace", " ", True),
        ("is_whitespace", "\t", True),
        ("is_whitespace", "a", False),
        ("is_parenthesis", "(", True),
        ("is_parenthesis", ")", True),
        ("is_parenthesis", "[", False),
        ("is_operator", "+", True),
        ("is_operator", "/", True),
        ("is_operator", "^", False),
        ("is_number_or_float", "7", True),
        ("is_number_or_float", ".", True),
        ("is_number_or_float", "x", False),
    ],
)
def test_character_classification(method, char, expected):
    assert getattr(Scanner(), method)(char) is expected


@pytest.mark.xfail(strict=True, reason="unknown characters are silently dropped")
def test_unknown_character_should_raise_error():
    with pytest.raises(CalcException):
        scan("2 a")


@pytest.mark.xfail(strict=True, reason="Scanner keeps tokens between scan_expr calls")
def test_scanner_can_be_reused_for_several_expressions():
    scanner = Scanner()
    scanner.scan_expr("1+2\n")
    assert len(scanner.scan_expr("3\n")) == 1