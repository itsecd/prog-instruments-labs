import pytest
from unittest.mock import Mock, patch
from lab_5.code import Product, Order, Shop

@pytest.mark.parametrize("initial_stock, reduce_by, expected_result, expected_stock", [
    (5, 2, True, 3),
    (5, 5, True, 0),
])
def test_product_reduce_stock_success(initial_stock, reduce_by, expected_result, expected_stock):
    product = Product("Laptop", 999.99, initial_stock)
    assert product.reduce_stock(reduce_by) == expected_result
    assert product.stock == expected_stock

@pytest.mark.parametrize("initial_stock, reduce_by, expected_result, expected_stock", [
    (5, 6, False, 5),
    (5, 10, False, 5),
])
def test_product_reduce_stock_failure(initial_stock, reduce_by, expected_result, expected_stock):
    product = Product("Laptop", 999.99, initial_stock)
    assert product.reduce_stock(reduce_by) == expected_result
    assert product.stock == expected_stock

def test_product_str():
    product = Product("Laptop", 999.99, 5)
    assert str(product) == "Laptop: $999.99 (5 in stock)"

def test_order_add_product():
    product = Product("Laptop", 999.99, 5)
    order = Order()
    order.add_product(product, 2)
    assert "Laptop" in order.items
    assert order.items["Laptop"]["quantity"] == 2

def test_order_calculate_total():
    product1 = Mock(spec=Product)
    product1.price = 999.99
    product2 = Mock(spec=Product)
    product2.price = 19.99

    order = Order()
    order.add_product = Mock()
    order.add_product(product1, 2)
    order.add_product(product2, 3)

    order.calculate_total = Mock(return_value=(2 * 999.99 + 3 * 19.99))
    total = order.calculate_total()
    assert total == (2 * 999.99 + 3 * 19.99)

def test_shop_add_product():
    shop = Shop()
    product = Product("Phone", 499.99, 10)
    shop.add_product(product)
    assert product in shop.products

def test_shop_find_product():
    shop = Shop()
    shop.add_product = Mock()
    shop.find_product = Mock(return_value="Mocked Phone")

    product = Product("Phone", 499.99, 10)
    shop.add_product(product)

    found_product = shop.find_product("Phone")
    assert found_product == "Mocked Phone"
    shop.find_product.assert_called_with("Phone")

def test_shop_find_product_not_found():
    shop = Shop()
    shop.find_product = Mock(return_value=None)
    found_product = shop.find_product("Laptop")
    assert found_product is None
    shop.find_product.assert_called_with("Laptop")
