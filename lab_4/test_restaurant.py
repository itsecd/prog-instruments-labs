"""Tests for restaurant.py."""

import unittest
from unittest.mock import patch

import restaurant


class TestCalculateBill(unittest.TestCase):
    def test_total_with_taxes(self):
        menu = {"Tea": 20, "Pizza": 100}
        order = {"Tea": 2, "Pizza": 1}
        total, gst, cgst, grand_total = restaurant.calculate_bill(order, menu)
        self.assertEqual(total, 140)
        self.assertAlmostEqual(gst, 7)
        self.assertAlmostEqual(cgst, 7)
        self.assertAlmostEqual(grand_total, 154)

    def test_empty_order(self):
        self.assertEqual(restaurant.calculate_bill({}, {"Tea": 20}), (0, 0, 0, 0))


class TestReadInt(unittest.TestCase):
    @patch("builtins.input", side_effect=["abc", "", "5"])
    def test_skips_not_numbers(self, _):
        self.assertEqual(restaurant.read_int("number: "), 5)

    @patch("builtins.input", side_effect=["0", "-3", "2"])
    def test_minimum(self, _):
        self.assertEqual(restaurant.read_int("number: ", 1), 2)


class TestMenu(unittest.TestCase):
    @patch("builtins.input", side_effect=["Tea", "20"])
    def test_add_item(self, _):
        menu = {}
        restaurant.add_item(menu)
        self.assertEqual(menu, {"Tea": 20})

    @patch("builtins.input", side_effect=["   "])
    def test_add_item_empty_name(self, _):
        menu = {}
        restaurant.add_item(menu)
        self.assertEqual(menu, {})

    @patch("builtins.input", side_effect=["Tea"])
    def test_delete_item(self, _):
        menu = {"Tea": 20, "Pizza": 100}
        restaurant.delete_item(menu)
        self.assertEqual(menu, {"Pizza": 100})

    @patch("builtins.input", side_effect=["Cake"])
    def test_delete_missing_item(self, _):
        menu = {"Tea": 20}
        restaurant.delete_item(menu)
        self.assertEqual(menu, {"Tea": 20})


class TestAccounts(unittest.TestCase):
    @patch("builtins.input", side_effect=["Ann", "ann", "pw", "ann", "pw"])
    def test_register_and_login(self, _):
        users = {}
        self.assertTrue(restaurant.register(users))
        self.assertEqual(users["ann"], {"name": "Ann", "password": "pw"})

    @patch("builtins.input", side_effect=["bob", "pw"])
    def test_login_unknown_user(self, _):
        self.assertFalse(restaurant.login({}))

    @patch("builtins.input", side_effect=["ann", "wrong", "ann", "pw"])
    def test_login_wrong_password_then_ok(self, _):
        users = {"ann": {"name": "Ann", "password": "pw"}}
        self.assertTrue(restaurant.login(users))


if __name__ == "__main__":
    unittest.main()
