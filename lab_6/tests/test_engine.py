"""Behavior-focused pytest tests for the micrograd Value class."""

from unittest.mock import Mock

import pytest

from lab_6.micrograd_lab import Value


@pytest.fixture
def operands():
    """A reusable pair of independent input nodes."""
    return Value(2.0), Value(3.0)


def test_constructor_initializes_value_and_gradient():
    node = Value(7)
    assert node.data == 7.0
    assert node.grad == 0.0
    assert node._prev == set()


def test_addition_between_nodes(operands):
    left, right = operands
    result = left + right
    assert result.data == 5.0
    assert result._prev == {left, right}


def test_multiplication_between_nodes(operands):
    left, right = operands
    assert (left * right).data == 6.0


def test_negation_and_subtraction(operands):
    left, right = operands
    assert (-left).data == -2.0
    assert (left - right).data == -1.0
    assert (10 - left).data == 8.0


def test_division_and_reverse_division(operands):
    left, right = operands
    assert (right / left).data == pytest.approx(1.5)
    assert (8 / left).data == pytest.approx(4.0)


def test_power_and_invalid_exponent():
    assert (Value(3) ** 2).data == 9.0
    with pytest.raises(TypeError, match="exponents"):
        Value(2).__pow__("not a number")


def test_reverse_arithmetic_with_plain_numbers():
    node = Value(2)
    assert (5 + node).data == 7.0
    assert (4 * node).data == 8.0
    assert (node + 4).data == 6.0
    assert (node * 3).data == 6.0


def test_repr_exposes_value_and_gradient():
    node = Value(1.5)
    assert repr(node) == "Value(data=1.5, grad=0.0)"


def test_gradient_for_composed_polynomial():
    # f(a, b) = ab + a²; df/da = b + 2a; df/db = a.
    a, b = Value(2), Value(-3)
    output = a * b + a**2
    output.backward()
    assert output.data == pytest.approx(-2)
    assert a.grad == pytest.approx(1)
    assert b.grad == pytest.approx(2)


def test_shared_node_accumulates_both_gradient_paths():
    x = Value(3)
    output = x * x + x
    output.backward()
    assert x.grad == pytest.approx(7)  # derivative of x² + x at x=3


@pytest.mark.parametrize(
    ("input_value", "expected_value", "expected_gradient"),
    [(-4.0, 0.0, 0.0), (-0.01, 0.0, 0.0), (0.0, 0.0, 0.0),
     (0.5, 0.5, 1.0), (9.0, 9.0, 1.0)],
    ids=["negative", "near-zero-negative", "zero", "fraction", "positive"],
)
def test_relu_cases(input_value, expected_value, expected_gradient):
    """Advanced test #1: parametrization of boundary and regular cases."""
    x = Value(input_value)
    output = x.relu()
    output.backward()
    assert output.data == pytest.approx(expected_value)
    assert x.grad == pytest.approx(expected_gradient)


@pytest.mark.parametrize(
    ("power", "input_value", "expected_gradient"),
    [(1, 5.0, 1.0), (2, -3.0, -6.0), (3, 2.0, 12.0), (-1, 2.0, -0.25)],
)
def test_power_gradients(power, input_value, expected_gradient):
    x = Value(input_value)
    (x**power).backward()
    assert x.grad == pytest.approx(expected_gradient)


def test_backward_invokes_each_unique_graph_node_once(monkeypatch):
    """Advanced test #2: spy on graph traversal via monkeypatch + Mock."""
    shared = Value(2)
    inner = shared * shared
    output = inner + inner  # two routes share the same predecessor
    original_backward = inner._backward
    spy = Mock(wraps=original_backward)
    monkeypatch.setattr(inner, "_backward", spy)

    output.backward()

    spy.assert_called_once_with()
    assert shared.grad == pytest.approx(8.0)


def test_independent_graphs_keep_gradients_separate():
    first = Value(2)
    second = Value(5)
    (first * 4).backward()
    assert first.grad == pytest.approx(4)
    assert second.grad == 0.0


def test_gradient_chain_rule_through_relu_and_product():
    a = Value(3)
    b = Value(2)
    output = (a * b - 4).relu()
    output.backward()
    assert output.data == pytest.approx(2.0)
    assert a.grad == pytest.approx(2.0)
    assert b.grad == pytest.approx(3.0)
