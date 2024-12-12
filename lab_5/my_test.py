import pytest
from menu import Menu, MenuItem
from order import Order
from utils import format_currency


def test_add_item_to_menu():
    menu = Menu()
    item = MenuItem("Кофе", 150)
    menu.add_item(item)
    assert len(menu.items) == 1
    assert menu.items[0].name == "Кофе"
    assert menu.items[0].price == 150


def test_remove_item_from_menu():
    menu = Menu()
    item = MenuItem("Чай", 100)
    menu.add_item(item)
    menu.remove_item("Чай")
    assert len(menu.items) == 0


def test_order_total_calculation():
    menu_item1 = MenuItem("Кофе", 150)
    menu_item2 = MenuItem("Сэндвич", 200)
    order = Order()
    order.add_to_order(menu_item1, 2)  # 2 * 150
    order.add_to_order(menu_item2, 1)  # 1 * 200
    assert order.calculate_total() == 500


def test_format_currency():
    assert format_currency(1234.5) == "1234.50 руб."


@pytest.mark.parametrize(
    "price,quantity,expected",
    [
        (150, 2, 300),  # 2 кофе по 150
        (200, 3, 600),  # 3 сэндвича по 200
        (100, 1, 100),  # 1 чай по 100
    ],
)
def test_order_total_with_parametrize(price, quantity, expected):
    item = MenuItem("Тестовый товар", price)
    order = Order()
    order.add_to_order(item, quantity)
    assert order.calculate_total() == expected


def test_menu_find_item_mock(mocker):
    menu = Menu()
    mocker.patch.object(menu, "find_item", return_value=MenuItem("Кофе", 150))
    item = menu.find_item("Кофе")
    assert item.name == "Кофе"
    assert item.price == 150


def test_menu_item_negative_price():
    with pytest.raises(ValueError, match="Цена не может быть отрицательной"):
        MenuItem("Ошибка", -50)


def test_invalid_currency_format():
    with pytest.raises(ValueError, match="Сумма не может быть отрицательной"):
        format_currency(-100)