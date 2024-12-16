import unittest
from lab_5.code import Product, Order, Shop

def test_product_reduce_stock_success():
    product = Product("Laptop", 999.99, 5)
    assert product.reduce_stock(2) is True
    assert product.stock == 3

def test_product_reduce_stock_failure():
    product = Product("Laptop", 999.99, 5)
    assert product.reduce_stock(6) is False
    assert product.stock == 5

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
    product1 = Product("Laptop", 999.99, 5)
    product2 = Product("Mouse", 19.99, 10)
    order = Order()
    order.add_product(product1, 2)
    order.add_product(product2, 3)
    total = order.calculate_total()
    assert total == (2 * 999.99 + 3 * 19.99)

def test_shop_add_product():
    shop = Shop()
    product = Product("Phone", 499.99, 10)
    shop.add_product(product)
    assert product in shop.products

def test_shop_find_product():
    shop = Shop()
    product = Product("Phone", 499.99, 10)
    shop.add_product(product)
    found_product = shop.find_product("Phone")
    assert found_product is product

def test_shop_find_product_not_found():
    shop = Shop()
    product = Product("Phone", 499.99, 10)
    shop.add_product(product)
    found_product = shop.find_product("Laptop")
    assert found_product is None