import pytest
from unittest.mock import Mock
from manageer import OrderManager

@pytest.fixture
def order_manager():
    return OrderManager()

def test_add_order_success(order_manager):
    order_manager.add_order(1, ["item1", "item2"], 100)
    assert len(order_manager.orders) == 1

def test_add_order_duplicate_id(order_manager):
    order_manager.add_order(1, ["item1"], 50)
    with pytest.raises(ValueError, match="Order ID already exists"):
        order_manager.add_order(1, ["item2"], 100)

def test_add_order_negative_total(order_manager):
    with pytest.raises(ValueError, match="Total cannot be negative"):
        order_manager.add_order(1, ["item1"], -50)

def test_get_order_success(order_manager):
    order_manager.add_order(1, ["item1"], 100)
    order = order_manager.get_order(1)
    assert order["total"] == 100
    assert "item1" in order["items"]

def test_get_order_not_found(order_manager):
    with pytest.raises(ValueError, match="Order not found"):
        order_manager.get_order(99)

def test_remove_order_success(order_manager):
    order_manager.add_order(1, ["item1"], 100)
    order_manager.remove_order(1)
    assert len(order_manager.orders) == 0

def test_remove_order_not_found(order_manager):
    with pytest.raises(ValueError, match="Order not found"):
        order_manager.remove_order(99)


@pytest.mark.parametrize("orders,expected_total", [
    ([{"order_id": 1, "items": ["item1"], "total": 50}], 50),
    ([{"order_id": 1, "items": ["item1"], "total": 50},
      {"order_id": 2, "items": ["item2"], "total": 100}], 150),
    ([], 0),
])
def test_calculate_total_sales(order_manager, orders, expected_total):
    order_manager.orders = orders
    assert order_manager.calculate_total_sales() == expected_total


def test_mock_add_order():
    mock_manager = Mock(spec=OrderManager)
    mock_manager.add_order.return_value = None
    mock_manager.add_order(1, ["item1"], 100)
    mock_manager.add_order.assert_called_once_with(1, ["item1"], 100)
