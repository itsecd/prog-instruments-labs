import json
import os


filename = "warehouse.json"
products = []

def save_products(products):
    try:
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(products, file, ensure_ascii=False, indent=4)

        print("Данные сохранены")
    except Exception:
        print("Ошибка при сохранении данных")

if os.path.exists(filename):
    try:
        file = open(filename, "r", encoding="utf-8")
        products = json.load(file)
        file.close()
        print("Данные загружены из файла")
    except Exception:
        print("Не получилось загрузить данные из файла")
        products = []


while True:
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

    choice = input("Выберите действие: ")

    if choice == "1":
        print("\nДобавление товара")
        name = input("Название: ").strip()

        if name == "":
            print("Название не может быть пустым")
        else:
            exists = False

            for product in products:
                if product["name"].lower() == name.lower():
                    exists = True

            if exists:
                print("Товар с таким названием уже существует")
            else:
                category = input("Категория: ").strip()

                try:
                    price = float(input("Цена за штуку: "))
                    quantity = int(input("Количество: "))

                    if price < 0 or quantity < 0:
                        print("Цена и количество не могут быть отрицательными")
                    else:
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

    elif choice == "2":
        print("\nСписок товаров")

        if len(products) == 0:
            print("На складе пока нет товаров")
        else:
            number = 1

            for product in products:
                print("Товар №", number)
                print("Название:", product["name"])
                print("Категория:", product["category"])
                print("Цена:", product["price"])
                print("Количество:", product["quantity"])
                print("Стоимость:", product["price"] * product["quantity"])
                number = number + 1

    elif choice == "3":
        print("\nПоиск товара")
        search = input("Введите название или его часть: ").strip().lower()

        if search == "":
            print("Поисковая строка не может быть пустой")
        else:
            found = False

            for product in products:
                if search in product["name"].lower():
                    print("Название:", product["name"])
                    print("Категория:", product["category"])
                    print("Цена:", product["price"])
                    print("Количество:", product["quantity"])
                    found = True

            if found == False:
                print("Ничего не найдено")

    elif choice == "4":
        print("\n Изменение количества")
        name = input("Введите название товара: ").strip().lower()

        found = False

        for product in products:
            if product["name"].lower() == name:
                found = True

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

                break

        if found == False:
            print("Товар не найден")

    elif choice == "5":
        print("\n Изменение цены")
        name = input("Введите название товара: ").strip().lower()

        found = False

        for product in products:
            if product["name"].lower() == name:
                found = True
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

                break

        if found == False:
            print("Товар не найден")

    elif choice == "6":
        print("\nУдаление товара")
        name = input("Введите название товара: ").strip().lower()

        found = False

        for product in products:
            if product["name"].lower() == name:
                found = True
                answer = input("Удалить товар? Да/нет: ").strip().lower()

                if answer == "да":
                    products.remove(product)
                    print("Товар удалён")
                else:
                    print("Удаление отменено")

                break

        if found == False:
            print("Товар не найден")

    elif choice == "7":
        print("\nТовары с маленьким остатком")

        try:
            limit = int(input("Показать товары с количеством меньше: "))

            if limit < 0:
                print("Значение не может быть отрицательным")
            else:
                found = False

                for product in products:
                    if product["quantity"] < limit:
                        
                        print("Название:", product["name"])
                        print("Категория:", product["category"])
                        print("Цена:", product["price"])
                        print("Количество:", product["quantity"])
                        found = True

                if found == False:
                    print("Таких товаров нет")

        except ValueError:
            print("Нужно ввести целое число")

    elif choice == "8":
        print("\nСтоимость товаров")

        if len(products) == 0:
            print("На складе пока нет товаров")
        else:
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

    elif choice == "9":
        save_products(products)

    elif choice == "0":
        save_products(products)
        print("Программа завершена")
        break

    else:
        print("Такого пункта меню нет")