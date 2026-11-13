import pytest

from src.token import (
    NumberToken,
    OperatorToken,
    ParenthesisToken,
    Token,
    TokenType,
    print_tokens,
)


def test_base_token_has_no_type():
    assert Token().type is None


@pytest.mark.parametrize("value", [0, 7, 3.5])
def test_number_token_keeps_value(value):
    token = NumberToken(value)
    assert token.type == TokenType.NUMBER
    assert token.value == value


@pytest.mark.parametrize(
    "operator, expected_type",
    [
        ("+", TokenType.PLUS),
        ("-", TokenType.MINUS),
        ("*", TokenType.TIMES),
        ("/", TokenType.DIV),
    ],
)
def test_operator_token_type(operator, expected_type):
    token = OperatorToken(operator)
    assert token.type == expected_type
    assert token.value == operator


def test_operator_token_rejects_unknown_operator():
    with pytest.raises(KeyError):
        OperatorToken("^").type


@pytest.mark.parametrize(
    "char, expected_type",
    [("(", TokenType.LPAREN), (")", TokenType.RPAREN)],
)
def test_parenthesis_token_type(char, expected_type):
    token = ParenthesisToken(char)
    assert token.type == expected_type
    assert token.value == char


def test_print_tokens_shows_every_token_and_expression(capsys):
    tokens = [
        ParenthesisToken("("),
        NumberToken(1),
        OperatorToken("+"),
        NumberToken(2),
        ParenthesisToken(")"),
    ]

    print_tokens(tokens)

    output = capsys.readouterr().out
    assert "Parenthesis:\t(" in output
    assert "Number:\t\t1" in output
    assert "Operator:\t+" in output
    assert output.strip().endswith("Expression:\t(1 + 2)")


def test_print_tokens_with_empty_list_prints_empty_expression(capsys):
    print_tokens([])

    assert "Expression:\t" in capsys.readouterr().out