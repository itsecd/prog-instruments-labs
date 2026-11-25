"""Scalar automatic differentiation adapted from Andrej Karpathy's micrograd.

Original: https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
MIT license, copyright (c) 2020 Andrej Karpathy.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable


class Value:
    """A scalar value in a computation graph, with its derivative."""

    def __init__(
        self,
        data: float,
        _children: Iterable[Value] = (),
        _op: str = "",
    ) -> None:
        self.data: float = float(data)
        self.grad: float = 0.0
        self._backward: Callable[[], None] = lambda: None
        self._prev: set[Value] = set(_children)
        self._op = _op

    def __add__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other: float) -> Value:
        if not isinstance(other, (int, float)):
            raise TypeError("Only integer and float exponents are supported")
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward() -> None:
            self.grad += other * self.data ** (other - 1) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> Value:
        out = Value(max(0.0, self.data), (self,), "ReLU")

        def _backward() -> None:
            self.grad += float(out.data > 0) * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        topo: list[Value] = []
        visited: set[Value] = set()

        def build_topo(node: Value) -> None:
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                topo.append(node)

        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __neg__(self) -> Value:
        return self * -1

    def __radd__(self, other: float) -> Value:
        return self + other

    def __sub__(self, other: Value | float) -> Value:
        return self + (-other if isinstance(other, Value) else -other)

    def __rsub__(self, other: float) -> Value:
        return -self + other

    def __rmul__(self, other: float) -> Value:
        return self * other

    def __truediv__(self, other: Value | float) -> Value:
        value = other if isinstance(other, Value) else Value(other)
        return self * value**-1

    def __rtruediv__(self, other: float) -> Value:
        return other * self**-1

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"
