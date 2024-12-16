class Product:
    def __init__(self, name, price, stock):
        self.name = name
        self.price = price
        self.stock = stock

    def __str__(self):
        return f"{self.name}: ${self.price:.2f} ({self.stock} in stock)"

    def reduce_stock(self, quantity):
        if quantity <= self.stock:
            self.stock -= quantity
            return True
        return False

class Order:
    def __init__(self):
        self.items = {}

    def add_product(self, product, quantity):
        if product.name in self.items:
            self.items[product.name]['quantity'] += quantity
        else:
            self.items[product.name] = {'product': product, 'quantity': quantity}

    def calculate_total(self):
        return sum(item['product'].price * item['quantity'] for item in self.items.values())

    def display_order(self):
        if not self.items:
            print("Your order is empty.")
        else:
            print("Your order:")
            for name, details in self.items.items():
                print(f"{name}: {details['quantity']} x ${details['product'].price:.2f}")

    def clear_order(self):
        self.items = {}

class Shop:
    def __init__(self):
        self.products = []

    def add_product(self, product):
        self.products.append(product)

    def display_products(self):
        if not self.products:
            print("No products available.")
        else:
            print("Available products:")
            for product in self.products:
                print(product)

    def find_product(self, name):
        for product in self.products:
            if product.name.lower() == name.lower():
                return product
        return None