class MenuItem:
    def __init__(self, name: str, price: float):
        if price < 0:
            raise ValueError("Цена не может быть отрицательной")
        self.name = name
        self.price = price

    def __str__(self):
        return f"{self.name}: {self.price:.2f} руб."

class Menu:
    def __init__(self):
        self.items = []

    def add_item(self, item: MenuItem):
        self.items.append(item)

    def remove_item(self, item_name: str):
        self.items = [item for item in self.items if item.name != item_name]

    def find_item(self, item_name: str) -> MenuItem:
        for item in self.items:
            if item.name == item_name:
                return item
        raise ValueError(f"Блюдо '{item_name}' не найдено")

    def get_menu(self):
        return [str(item) for item in self.items]
