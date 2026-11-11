import json
import os


FILENAME = "warehouse.json"

def save_products(products):
    try:
        with open(FILENAME, "w", encoding="utf-8") as file:
            json.dump(products, file, ensure_ascii=False, indent=4)

        print("Данные сохранены")
    except Exception:
        print("Ошибка при сохранении данных")

def load_products():
    if os.path.exists(FILENAME):
        try:
            with open(FILENAME, "r", encoding="utf-8") as file:
                loaded_products = json.load(file)

            print("Данные загружены из файла")
            return loaded_products

        except Exception:
            print("Не получилось загрузить данные из файла")
            return []

    return []

def print_product(product):
    print("Название:", product["name"])
    print("Категория:", product["category"])
    print("Цена:", product["price"])
    print("Количество:", product["quantity"])

def find_product(products, name):
    for product in products:
        if product["name"].lower() == name.lower():
            return product

    return None


def add_product(products):
    print("\nДобавление товара")
    name = input("Название: ").strip()

    if name == "":
        print("Название не может быть пустым")
        return

    if find_product(products, name) is not None:
        print("Товар с таким названием уже существует")
        return

    category = input("Категория: ").strip()

    try:
        price = float(input("Цена за штуку: "))
        quantity = int(input("Количество: "))

        if price < 0 or quantity < 0:
            print("Цена и количество не могут быть отрицательными")
            return

        product = {
            "name": name,
            "category": category,
            "price": price,
            "quantity": quantity
        }

        products.append(product)
        print("Товар добавлен")

    except ValueError:
        print("Ошибка: цена и количество должны быть числами")


def show_products(products):
    print("\nСписок товаров")

    if len(products) == 0:
        print("На складе пока нет товаров")
        return

    number = 1

    for product in products:
        print("Товар №", number)
        print_product(product)
        print("Стоимость:", product["price"] * product["quantity"])
        number = number + 1


def search_products(products):
    print("\nПоиск товара")
    search = input("Введите название или его часть: ").strip().lower()

    if search == "":
        print("Поисковая строка не может быть пустой")
        return

    found = False

    for product in products:
        if search in product["name"].lower():
            print_product(product)
            found = True

    if found == False:
        print("Ничего не найдено")

def change_quantity(products):
    print("\n Изменение количества")
    name = input("Введите название товара: ").strip().lower()
    product = find_product(products, name)

    if product is None:
        print("Товар не найден")
        return

    print("Текущее количество:", product["quantity"])
    print("1. Добавить товар")
    print("2. Списать товар")
    print("3. Установить новое количество")

    action = input("Выберите действие: ")

    if action == "1":
        try:
            amount = int(input("Сколько добавить: "))

            if amount < 0:
                print("Количество не может быть отрицательным")
            else:
                product["quantity"] = product["quantity"] + amount
                print("Новое количество:", product["quantity"])

        except ValueError:
            print("Нужно ввести целое число")

    elif action == "2":
        try:
            amount = int(input("Сколько списать: "))

            if amount < 0:
                print("Количество не может быть отрицательным")
            elif amount > product["quantity"]:
                print("На складе недостаточно товара")
            else:
                product["quantity"] = product["quantity"] - amount
                print("Новое количество:", product["quantity"])

        except ValueError:
            print("Нужно ввести целое число")

    elif action == "3":
        try:
            amount = int(input("Введите новое количество: "))

            if amount < 0:
                print("Количество не может быть отрицательным")
            else:
                product["quantity"] = amount
                print("Количество изменено")

        except ValueError:
            print("Нужно ввести целое число")

    else:
        print("Такого действия нет")


def change_price(products):
    print("\n Изменение цены")
    name = input("Введите название товара: ").strip().lower()
    product = find_product(products, name)

    if product is None:
        print("Товар не найден")
        return

    print("Текущая цена:", product["price"])

    try:
        new_price = float(input("Введите новую цену: "))

        if new_price < 0:
            print("Цена не может быть отрицательной")
        else:
            product["price"] = new_price
            print("Цена изменена")

    except ValueError:
        print("Нужно ввести число")


def delete_product(products):
    print("\nУдаление товара")
    name = input("Введите название товара: ").strip().lower()
    product = find_product(products, name)

    if product is None:
        print("Товар не найден")
        return

    answer = input("Удалить товар?").strip().lower()

    if answer == "да":
        products.remove(product)
        print("Товар удалён")
    else:
        print("Удаление отменено")

def show_low_stock(products):
    print("\nТовары с маленьким остатком")

    try:
        limit = int(input("Показать товары с количеством меньше: "))

        if limit < 0:
            print("Значение не может быть отрицательным")
        else:
            found = False

            for product in products:
                if product["quantity"] < limit:
                    print_product(product)
                    found = True

            if found == False:
                print("Таких товаров нет")

    except ValueError:
        print("Нужно ввести целое число")


def show_cost(products):
    print("\nСтоимость товаров")

    if len(products) == 0:
        print("На складе пока нет товаров")
        return

    total = 0
    total_quantity = 0

    for product in products:
        product_cost = product["price"] * product["quantity"]
        total = total + product_cost
        total_quantity = total_quantity + product["quantity"]

        print(
            product["name"],
            "-",
            product["quantity"],
            "шт. на сумму",
            round(product_cost, 2)
        )

    print("Всего видов товаров:", len(products))
    print("Всего единиц товаров:", total_quantity)
    print("Общая стоимость:", round(total, 2))


def print_menu():
    print("1. Добавить товар")
    print("2. Показать все товары")
    print("3. Найти товар по названию")
    print("4. Изменить количество товара")
    print("5. Изменить цену товара")
    print("6. Удалить товар")
    print("7. Показать товары, которых мало")
    print("8. Показать общую стоимость товаров")
    print("9. Сохранить данные")
    print("0. Сохранить и выйти")

def main():
    products = load_products()

    while True:
        print_menu()
        choice = input("Выберите действие: ")

        if choice == "1":
            add_product(products)

        elif choice == "2":
            show_products(products)

        elif choice == "3":
            search_products(products)

        elif choice == "4":
            change_quantity(products)

        elif choice == "5":
            change_price(products)

        elif choice == "6":
            delete_product(products)

        elif choice == "7":
            show_low_stock(products)

        elif choice == "8":
            show_cost(products)

        elif choice == "9":
            save_products(products)

        elif choice == "0":
            save_products(products)
            print("Программа завершена")
            break

        else:
            print("Такого пункта тут нет :)")


if __name__ == "__main__":
    main()