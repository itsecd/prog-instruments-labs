from menu import MenuItem

class Order:
    def __init__(self):
        self.items = []

    def add_to_order(self, item: MenuItem, quantity: int):
        if quantity <= 0:
            raise ValueError("Количество должно быть больше нуля")
        self.items.append((item, quantity))

    def remove_from_order(self, item_name: str):
        self.items = [entry for entry in self.items if entry[0].name != item_name]

    def calculate_total(self) -> float:
        return sum(item.price * quantity for item, quantity in self.items)

    def list_order(self):
        return [f"{item.name} x{quantity}" for item, quantity in self.items]
