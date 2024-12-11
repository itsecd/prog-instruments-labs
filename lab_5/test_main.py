import pytest
from unittest.mock import Mock
from lab_5.main import Product, ShoppingCart, Discount


def test_product_total_price():
    product = Product("Apple", 10, 3)
    assert product.total_price() == 30

def test_product_equality():
    product1 = Product("Apple", 10, 3)
    product2 = Product("Apple", 10, 3)
    assert product1 == product2

def test_product_negative_values():
    with pytest.raises(ValueError):
        Product("Apple", -5, 1)
    with pytest.raises(ValueError):
        Product("Apple", 5, -1)

def test_cart_add_product():
    cart = ShoppingCart()
    product = Product("Apple", 10)
    cart.add_product(product)
    assert len(cart.items) == 1
    assert cart.items[0] == product

def test_cart_remove_product():
    cart = ShoppingCart()
    product = Product("Apple", 10)
    cart.add_product(product)
    cart.remove_product("Apple")
    assert len(cart.items) == 0

def test_cart_remove_nonexistent_product():
    cart = ShoppingCart()
    cart.add_product(Product("Apple", 10))
    cart.remove_product("Banana")
    assert len(cart.items) == 1

def test_cart_group_by_category():
    cart = ShoppingCart()
    cart.add_product(Product("Apple", 10, category="Fruits"))
    cart.add_product(Product("Banana", 5, category="Fruits"))
    cart.add_product(Product("Carrot", 3, category="Vegetables"))
    grouped = cart.group_by_category()
    assert "Fruits" in grouped
    assert len(grouped["Fruits"]) == 2
    assert "Vegetables" in grouped
    assert len(grouped["Vegetables"]) == 1

def test_cart_clear():
    cart = ShoppingCart()
    cart.add_product(Product("Apple", 10))
    cart.clear()
    assert len(cart.items) == 0

@pytest.mark.parametrize("total, discount, expected", [
    (100, 10, 90),
    (200, 25, 150),
    (50, 0, 50),
    (100, 100, 0),
])
def test_apply_discount(total, discount, expected):
    assert Discount.apply_discount(total, discount) == expected

def test_apply_coupons():
    coupons = [10, 20]
    total = 200
    final_total = Discount.apply_coupons(total, coupons)
    assert final_total == pytest.approx(144, 0.01)

def test_bulk_discount_mock():
    cart = ShoppingCart()
    cart.get_total_price = Mock(return_value=300)
    result = Discount.bulk_discount(cart, threshold=200, discount_percentage=20)
    assert result == 240

def test_bulk_discount_no_threshold():
    cart = ShoppingCart()
    cart.add_product(Product("Apple", 10, 5))
    result = Discount.bulk_discount(cart, threshold=100, discount_percentage=10)
    assert result == 50

def test_bulk_discount_below_threshold():
    cart = ShoppingCart()
    cart.add_product(Product("Apple", 10, 2))
    result = Discount.bulk_discount(cart, threshold=50, discount_percentage=10)
    assert result == 20

def test_cart_with_discount():
    cart = ShoppingCart()
    cart.add_product(Product("Apple", 10, 5))
    cart.add_product(Product("Banana", 15, 2))
    total = cart.get_total_price()
    discounted_total = Discount.apply_discount(total, 20)
    assert total == 80
    assert discounted_total == 64
