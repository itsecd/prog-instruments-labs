from src.ast import ExprTree, NumberTree, OperatorTree


def test_expr_tree_defaults_to_empty_node():
    tree = ExprTree()
    assert (tree.data, tree.lhs, tree.rhs) == (None, None, None)


def test_number_tree_exposes_number():
    tree = NumberTree(3.5)
    assert tree.number == 3.5
    assert tree.lhs is None and tree.rhs is None


def test_operator_tree_keeps_operator_and_operands():
    left, right = NumberTree(1), NumberTree(2)
    tree = OperatorTree("+", left, right)
    assert tree.operator == "+"
    assert tree.lhs is left
    assert tree.rhs is right