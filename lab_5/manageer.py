class OrderManager:
    def __init__(self):
        self.orders = []

    def add_order(self, order_id, items, total):
        if any(order["order_id"] == order_id for order in self.orders):
            raise ValueError("Order ID already exists")
        if total < 0:
            raise ValueError("Total cannot be negative")
        self.orders.append({"order_id": order_id, "items": items, "total": total})

    def get_order(self, order_id):
        for order in self.orders:
            if order["order_id"] == order_id:
                return order
        raise ValueError("Order not found")

    def remove_order(self, order_id):
        for order in self.orders:
            if order["order_id"] == order_id:
                self.orders.remove(order)
                return
        raise ValueError("Order not found")

    def calculate_total_sales(self):
        return sum(order["total"] for order in self.orders)
