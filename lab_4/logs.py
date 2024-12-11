import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='store.log',
                    filemode='w',
                    encoding='utf-8')


class Product:
    def __init__(self, name, price):
        self.name = name
        self.price = price

    def __str__(self):
        return f"{self.name} - {self.price} у.е."


class ShoppingCart:
    def __init__(self):
        self.products = []

    def add_product(self, product):
        logging.info(f"Добавление продукта: {product.name} с ценой {product.price}")
        self.products.append(product)

    def remove_product(self, product_name):
        for product in self.products:
            if product.name == product_name:
                logging.info(f"Удаление продукта: {product_name}")
                self.products.remove(product)
                return
        logging.warning(f"Продукт {product_name} не найден в корзине.")

    def calculate_total(self):
        total = sum(product.price for product in self.products)
        logging.info(f"Общая стоимость корзины: {total} у.е.")
        return total

    def show_products(self):
        if not self.products:
            logging.info("Корзина пуста.")
        else:
            logging.info("Продукты в корзине:")
            for product in self.products:
                logging.info(f"- {product}")

    def clear_cart(self):
        logging.info("Очистка корзины...")
        self.products.clear()
        logging.info("Корзина очищена.")

    def checkout(self):
        if not self.products:
            logging.warning("Корзина пуста. Попытка оформить заказ.")
            print("Корзина пуста. Добавьте продукты для оформления заказа.")
            return
        total = self.calculate_total()
        logging.info(f"Оформление заказа на сумму {total} у.е.")
        self.clear_cart()  # Очистить корзину после оформления заказа


class Store:
    def __init__(self):
        self.products = [
            Product("Хлеб", 1.0),
            Product("Молоко", 0.9),
            Product("Яблоки", 1.5),
            Product("Яйца", 2.0),
            Product("Кофе", 3.0),
            Product("Чай", 2.5)
        ]

    def display_products(self):
        logging.info("Показ доступных продуктов.")
        print("Доступные продукты:")
        for product in self.products:
            print(f"- {product}")


def main():
    store = Store()
    cart = ShoppingCart()

    while True:
        print("\nМеню:")
        print("1. Показать доступные продукты")
        print("2. Добавить продукт в корзину")
        print("3. Удалить продукт из корзины")
        print("4. Показать продукты в корзине")
        print("5. Оформить заказ")
        print("0. Выход")

        choice = input("Выберите опцию: ")

        if choice == '1':
            store.display_products()
        elif choice == '2':
            product_name = input("Введите название продукта для добавления: ")
            found = False
            for product in store.products:
                if product.name.lower() == product_name.lower():
                    cart.add_product(product)
                    found = True
                    break
            if not found:
                logging.warning(f"Продукт {product_name} не найден в магазине.")
                print(f"Продукт {product_name} не найден.")
        elif choice == '3':
            product_name = input("Введите название продукта для удаления: ")
            cart.remove_product(product_name)
        elif choice == '4':
            cart.show_products()
        elif choice == '5':
            cart.checkout()
        elif choice == '0':
            logging.info("Выход из программы.")
            print("Выход из программы.")
            break
        else:
            logging.error("Неверный ввод, выбрана недопустимая опция.")
            print("Неверный ввод, пожалуйста, выберите снова.")


if __name__ == "__main__":
    main()
