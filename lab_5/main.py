class Product:
    def __init__(self, name: str, price: float, quantity: int = 1, category: str = "General"):
        if price < 0 or quantity < 0:
            raise ValueError("Price and quantity must be non-negative")
        self.name = name
        self.price = price
        self.quantity = quantity
        self.category = category

    def total_price(self) -> float:
        return self.price * self.quantity

    def __eq__(self, other):
        if not isinstance(other, Product):
            return False
        return self.name == other.name and self.price == other.price and self.category == other.category


class ShoppingCart:
    def __init__(self, allowed_categories=None):
        self.items = []
        self.allowed_categories = allowed_categories or []

    def add_product(self, product: Product):
        if self.allowed_categories and product.category not in self.allowed_categories:
            raise ValueError(f"Category '{product.category}' is not allowed in this cart")
        self.items.append(product)

    def remove_product(self, product_name: str):
        self.items = [item for item in self.items if item.name != product_name]

    def get_total_price(self) -> float:
        return sum(item.total_price() for item in self.items)

    def clear(self):
        self.items = []

    def group_by_category(self):
        grouped = {}
        for item in self.items:
            if item.category not in grouped:
                grouped[item.category] = []
            grouped[item.category].append(item)
        return grouped


class Discount:
    @staticmethod
    def apply_discount(total: float, discount_percentage: float) -> float:
        if not (0 <= discount_percentage <= 100):
            raise ValueError("Discount percentage must be between 0 and 100")
        return total * (1 - discount_percentage / 100)

    @staticmethod
    def bulk_discount(cart: ShoppingCart, threshold: float, discount_percentage: float) -> float:
        total = cart.get_total_price()
        if total > threshold:
            return Discount.apply_discount(total, discount_percentage)
        return total

    @staticmethod
    def apply_coupons(total: float, coupons: list):
        for coupon in coupons:
            if not (0 <= coupon <= 100):
                raise ValueError("Coupon value must be between 0 and 100")
            total = total * (1 - coupon / 100)
        return total
