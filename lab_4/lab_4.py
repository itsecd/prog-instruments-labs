import json
import os
import random
import sys
import time


MONEY = 100.0
DAY = 1
INVENTORY = {}
FARM = []
WEATHER = "sunny"
PLAYER_NAME = "Farmer"
GAME_OVER = False
MESSAGE = ""
SAVE_FILE = "bad_farm_save.json"
PLANTS = [
    {"id": 1, "name": "Carrot", "grow": 3, "water_need": 1, "price": 2.5},
    {"id": 2, "name": "Potato", "grow": 5, "water_need": 2, "price": 4.0},
    {"id": 3, "name": "Tomato", "grow": 4, "water_need": 2, "price": 5.0},
    {"id": 4, "name": "Corn", "grow": 7, "water_need": 3, "price": 7.5},
    {"id": 5, "name": "Pumpkin", "grow": 10, "water_need": 4, "price": 15.0},
]
MAX_PLOTS = 6


def get_plant_by_id(plant_id):
    for plant in PLANTS:
        if plant["id"] == plant_id:
            return plant
    return None


def init_farm():
    global FARM, INVENTORY, MONEY, DAY, PLAYER_NAME
    FARM = []
    INVENTORY = {}
    for i in range(MAX_PLOTS):
        FARM.append({
            "plot": i,
            "state": "empty",
            "plant": None,
            "age": 0,
            "watered": 0
        })
    MONEY = 100.0
    DAY = 1
    PLAYER_NAME = "Farmer"


def get_plot_display_string(plot_data):
    plot_num = plot_data["plot"]
    state = plot_data["state"]

    display_string = f"[{plot_num}]"

    if state == "empty":
        display_string += " Пусто"
    elif state == "planted":
        plant = plot_data["plant"]
        display_string += f" {plant['name']} ({plot_data['age']}/{plant['grow']}) вода:{plot_data['watered']}/{plant['water_need']}"
    elif state == "ready":
        plant = plot_data["plant"]
        display_string += f" {plant['name']} (Готов)"
    elif state == "withered":
        display_string += " Завяли"
    else:
        display_string += " ???"

    return display_string


def show_farm():
    global FARM, DAY, MONEY, WEATHER, MESSAGE, INVENTORY
    print("\n" + "=" * 40)
    print(f"День: {DAY} | Деньги: ${MONEY:.2f} | Погода: {WEATHER}")
    print("Инвентарь:", INVENTORY)
    print("-" * 40)

    for plot in FARM:
        print(get_plot_display_string(plot))

    print("-" * 40)
    if MESSAGE:
        print(">>>", MESSAGE)
    print("=" * 40)


def get_valid_input(prompt, input_type=int, validation_func=None):
    try:
        value = input_type(input(prompt))
        if validation_func and not validation_func(value):
            return None
        return value
    except (ValueError, TypeError):
        return None


def plant_seed():
    global FARM, MESSAGE, MONEY

    plant_id = get_valid_input(
        "Введите ID растения для посадки (1-5): ",
        int,
        lambda x: 1 <= x <= 5
    )
    if plant_id is None:
        MESSAGE = "Неправильный ID."
        return

    plant = get_plant_by_id(plant_id)
    if not plant:
        MESSAGE = "Неправильный ID."
        return

    plot = get_valid_input(
        f"В какую грядку (0-{MAX_PLOTS - 1}): ",
        int,
        lambda x: 0 <= x < MAX_PLOTS
    )
    if plot is None:
        MESSAGE = "Неправильная грядка."
        return

    if FARM[plot]["state"] != "empty":
        MESSAGE = "Грядка уже занята."
        return

    cost = plant["price"] * 0.5 + plant["grow"] * 0.1
    if MONEY < cost:
        MESSAGE = f"Не хватает денег на семена (${cost:.2f} нужно)."
        return

    MONEY -= cost
    FARM[plot]["state"] = "planted"
    FARM[plot]["plant"] = plant
    FARM[plot]["age"] = 0
    FARM[plot]["watered"] = 0
    MESSAGE = f"Посадили {plant['name']} в грядку {plot} (стоимость ${cost:.2f})."


def water_plot():
    global FARM, MESSAGE
    plot = get_valid_input(
        f"Какую грядку полить (0-{MAX_PLOTS - 1}): ",
        int,
        lambda x: 0 <= x < MAX_PLOTS
    )
    if plot is None:
        MESSAGE = "Неправильная грядка."
        return

    if FARM[plot]["state"] not in ("planted", "ready"):
        MESSAGE = "Там нечего поливать."
        return

    FARM[plot]["watered"] += 1
    MESSAGE = f"Полили грядку {plot}."


def harvest_plot():
    global FARM, MONEY, INVENTORY, MESSAGE
    plot = get_valid_input(
        f"Какую грядку собрать (0-{MAX_PLOTS - 1}): ",
        int,
        lambda x: 0 <= x < MAX_PLOTS
    )
    if plot is None:
        MESSAGE = "Неправильная грядка."
        return

    if FARM[plot]["state"] != "ready":
        MESSAGE = "Там ещё не готово."
        return

    plant = FARM[plot]["plant"]
    amount = random.randint(1, 3) + int(plant["grow"] / 4)
    name = plant["name"]
    INVENTORY[name] = INVENTORY.get(name, 0) + amount
    FARM[plot]["state"] = "empty"
    FARM[plot]["plant"] = None
    FARM[plot]["age"] = 0
    FARM[plot]["watered"] = 0
    MESSAGE = f"Собрали {name} x{amount}."

    if random.random() < 0.05:
        MESSAGE += " Лопата сломалась (ничего не делает)."


def sell_items():
    global INVENTORY, MONEY, MESSAGE
    print("Инвентарь:", INVENTORY)
    item = input("Что продать (название) или оставить пустым: ").strip()

    if item == "":
        MESSAGE = "Ничего не продали."
        return

    if item not in INVENTORY or INVENTORY[item] <= 0:
        MESSAGE = "У вас нет такого товара."
        return

    try:
        count = int(input("Сколько продать (количество): "))
    except ValueError:
        count = 0

    if count <= 0 or count > INVENTORY[item]:
        MESSAGE = "Неправильное количество."
        return

    base_price = 0.0
    for plant in PLANTS:
        if plant["name"] == item:
            base_price = plant["price"]
            break

    price = base_price * (0.8 + (DAY % 5) * 0.05)
    MONEY += price * count
    INVENTORY[item] -= count

    if INVENTORY[item] == 0:
        del INVENTORY[item]

    MESSAGE = f"Продали {item} x{count} за ${price * count:.2f}."


def rest_day():
    global DAY, FARM, WEATHER, MESSAGE
    DAY += 1

    random_value = random.random()
    if random_value < 0.6:
        WEATHER = "sunny"
    elif random_value < 0.85:
        WEATHER = "rain"
    else:
        WEATHER = "storm"

    for farm_plot in FARM:
        try:
            if farm_plot["state"] == "planted":
                if WEATHER == "rain":
                    farm_plot["watered"] += 2
                if WEATHER == "storm":
                    if random.random() < 0.2:
                        farm_plot["state"] = "withered"
                        farm_plot["plant"] = None
                        farm_plot["age"] = 0
                        farm_plot["watered"] = 0
                        continue
                farm_plot["age"] += 1
                plant = farm_plot["plant"]

                if (farm_plot["age"] >= plant["grow"] and
                        farm_plot["watered"] >= plant["water_need"]):
                    farm_plot["state"] = "ready"
                elif (farm_plot["age"] >= plant["grow"] and
                      farm_plot["watered"] < plant["water_need"]):
                    farm_plot["state"] = "withered"

                if WEATHER == "sunny":
                    farm_plot["watered"] = max(0, farm_plot["watered"] - 1)

            elif farm_plot["state"] == "withered":
                if random.random() < 0.05:
                    farm_plot["state"] = "empty"
                    farm_plot["plant"] = None
        except Exception:
            pass

    MESSAGE = "Прошёл день."


def calculate_seed_cost(plant, use_quick_calculation=True):
    if use_quick_calculation:
        return plant["price"] * 0.5 + plant["grow"] * 0.1
    else:
        return plant["price"] * 0.5


def buy_seeds(use_quick_calculation=True):
    global MONEY, MESSAGE

    if use_quick_calculation:
        print("Семена в магазине:")
        for plant in PLANTS:
            cost = calculate_seed_cost(plant, True)
            print(plant["id"], plant["name"], "цена семян ~", cost)
        prompt_id = "ID семян: "
        prompt_count = "Сколько семян купить: "
    else:
        print("Добро пожаловать в магазин семян.")
        for plant in PLANTS:
            cost = calculate_seed_cost(plant, False)
            print(f"{plant['id']} - {plant['name']} - ${cost:.2f}")
        prompt_id = "Введите ID: "
        prompt_count = "Кол-во: "

    plant_id = get_valid_input(prompt_id, int)
    if plant_id is None:
        MESSAGE = "Не выбран тип семян." if use_quick_calculation else "Неверный выбор."
        return

    plant = get_plant_by_id(plant_id)
    if not plant:
        MESSAGE = "Не выбран тип семян." if use_quick_calculation else "Неверный выбор."
        return

    try:
        count = int(input(prompt_count))
    except ValueError:
        count = 0

    cost = calculate_seed_cost(plant, use_quick_calculation) * count

    if count <= 0:
        MESSAGE = "Ничего не купили."
        return

    if MONEY < cost:
        MESSAGE = "Недостаточно денег." if use_quick_calculation else "Не хватает денег."
        return

    MONEY -= cost
    name = plant["name"]
    INVENTORY[name + "_seed"] = INVENTORY.get(name + "_seed", 0) + count
    MESSAGE = f"Купили {name} x{count} семян."


def buy_seeds_quick():
    buy_seeds(use_quick_calculation=True)


def buy_seeds_shop():
    buy_seeds(use_quick_calculation=False)


def plant_from_inventory():
    global INVENTORY, FARM, MESSAGE

    seeds = {k: v for k, v in INVENTORY.items() if k.endswith("_seed")}
    print("Ваши семена:", seeds)

    seed_name = input("Введите название семени (например Carrot) или пусто: ").strip()
    if seed_name == "":
        MESSAGE = "Отменено."
        return

    seed_key = seed_name + "_seed"
    if INVENTORY.get(seed_key, 0) <= 0:
        MESSAGE = "Нет таких семян."
        return

    plot = get_valid_input(
        f"Грядка (0-{MAX_PLOTS - 1}): ",
        int,
        lambda x: 0 <= x < MAX_PLOTS
    )
    if plot is None:
        MESSAGE = "Неправильная грядка."
        return

    if FARM[plot]["state"] != "empty":
        MESSAGE = "Грядка занята."
        return

    plant = None
    for plant_item in PLANTS:
        if plant_item["name"].lower() == seed_name.lower():
            plant = plant_item
            break

    if not plant:
        MESSAGE = "Растение неизвестно."
        return

    INVENTORY[seed_key] -= 1
    if INVENTORY[seed_key] == 0:
        del INVENTORY[seed_key]

    FARM[plot]["state"] = "planted"
    FARM[plot]["plant"] = plant
    FARM[plot]["age"] = 0
    FARM[plot]["watered"] = 0
    MESSAGE = f"Посадили семя {plant['name']}."


def save_game():
    global SAVE_FILE, DAY, MONEY, INVENTORY, FARM, PLAYER_NAME, WEATHER, MESSAGE
    data = {
        "day": DAY,
        "money": MONEY,
        "invent": INVENTORY,
        "farm": FARM,
        "player": PLAYER_NAME,
        "weather": WEATHER,
    }
    try:
        with open(SAVE_FILE, "w", encoding="utf-8") as file:
            json.dump(data, file)
        MESSAGE = "Игра сохранена."
    except Exception as error:
        MESSAGE = f"Ошибка сохранения: {error}"


def load_game():
    global SAVE_FILE, DAY, MONEY, INVENTORY, FARM, PLAYER_NAME, WEATHER, MESSAGE
    if not os.path.exists(SAVE_FILE):
        MESSAGE = "Файл сохранения не найден."
        return

    try:
        with open(SAVE_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
        DAY = data.get("day", 1)
        MONEY = data.get("money", 100.0)
        INVENTORY = data.get("invent", {})
        FARM = data.get("farm", FARM)
        PLAYER_NAME = data.get("player", PLAYER_NAME)
        WEATHER = data.get("weather", "sunny")
        MESSAGE = "Игра загружена."
    except Exception:
        MESSAGE = "Ошибка загрузки."


def random_events():
    global MONEY, INVENTORY, MESSAGE
    random_value = random.random()

    if random_value < 0.1:
        lost = int(MONEY * 0.1)
        MONEY -= lost
        MESSAGE = f"Ночной вор унес ${lost}."
    elif random_value < 0.2:
        plant = random.choice(PLANTS)
        INVENTORY[plant["name"] + "_seed"] = INVENTORY.get(plant["name"] + "_seed", 0) + 1
        MESSAGE = f"Нашли семя {plant['name']} на дороге."
    elif random_value < 0.25:
        bonus = 10
        MONEY += bonus
        MESSAGE = f"Посетили ярмарку, получили бонус ${bonus}."
    else:
        MESSAGE = "Тихий день."


def market_prices():
    global PLANTS, DAY
    print("Рыночные цены (меняются):")
    for plant in PLANTS:
        price = plant["price"] * (0.9 + (DAY % 7) * 0.02)
        print(plant["name"], f"${price:.2f}")


def stats():
    global DAY, MONEY, INVENTORY, FARM
    total_planted = sum(1 for farm_plot in FARM if farm_plot["state"] in ("planted", "ready"))
    total_ready = sum(1 for farm_plot in FARM if farm_plot["state"] == "ready")
    print(f"Статистика --- День {DAY}. Деньги: ${MONEY:.2f}")
    print("Посажено:", total_planted, "Грядок готово:", total_ready, "Инвентарь:", INVENTORY)


def help_menu():
    print("""
Доступные команды:
1 - Показать ферму
2 - Посадить семена из магазина
3 - Купить семена (быстрая покупка)
4 - Посадить из инвентаря
5 - Полить грядку
6 - Собрать урожай
7 - Продать товар
8 - Следующий день (отдых)
9 - Сохранить игру
10 - Загрузить игру
11 - Рыночные цены
12 - Статистика
13 - Выйти
h - Помощь
""")


def handle_choice(choice):
    global GAME_OVER, MESSAGE
    if choice == "1":
        show_farm()
    elif choice == "2":
        plant_seed()
    elif choice == "3":
        buy_seeds_quick()
    elif choice == "shop":
        buy_seeds_shop()
    elif choice == "4":
        plant_from_inventory()
    elif choice == "5":
        water_plot()
    elif choice == "6":
        harvest_plot()
    elif choice == "7":
        sell_items()
    elif choice == "8":
        rest_day()
        random_events()
    elif choice == "9":
        save_game()
    elif choice == "10":
        load_game()
    elif choice == "11":
        market_prices()
    elif choice == "12":
        stats()
    elif choice == "13":
        GAME_OVER = True
    elif choice == "h":
        help_menu()
    else:
        MESSAGE = "Неизвестная команда."


def main_loop():
    global GAME_OVER, MESSAGE, PLAYER_NAME, MONEY
    print("Добро пожаловать на плохую ферму!")

    try:
        name = input(f"Введите имя фермера (Enter для '{PLAYER_NAME}'): ").strip()
        if name != "":
            PLAYER_NAME = name
    except Exception:
        pass

    MESSAGE = "Начинаем!"
    count = 0

    while not GAME_OVER:
        if count % 5 == 0:
            random_value = random.random()
            if random_value < 0.5:
                pass
            else:
                pass

        show_farm()
        print("Команды: 1..13, h - помощь")
        user_choice = input("Ваш выбор: ").strip()
        handle_choice(user_choice)

        if MONEY < 5 and INVENTORY:
            try:
                item = next(iter(INVENTORY))
                if item.endswith("_seed"):
                    pass
                else:
                    sold = 1
                    item_price = next((plant["price"] for plant in PLANTS if plant["name"] == item), 1.0)
                    MONEY += sold * item_price
                    INVENTORY[item] -= sold
                    if INVENTORY[item] <= 0:
                        del INVENTORY[item]
                    MESSAGE = f"Автопродажа: {item} x{sold}."
            except Exception:
                pass

        count += 1

        try:
            time.sleep(0.2)
        except Exception:
            pass


def autopilot(turns=20):
    global MESSAGE
    for i in range(turns):
        action = random.choice(["plant", "buy", "water", "rest", "harvest", "sell"])

        if action == "plant":
            for idx, farm_plot in enumerate(FARM):
                if farm_plot["state"] == "empty":
                    plant = random.choice(PLANTS)
                    FARM[idx]["state"] = "planted"
                    FARM[idx]["plant"] = plant
                    FARM[idx]["age"] = 0
                    FARM[idx]["watered"] = 0
                    break
        elif action == "buy":
            plant = random.choice(PLANTS)
            INVENTORY[plant["name"] + "_seed"] = INVENTORY.get(plant["name"] + "_seed", 0) + 1
        elif action == "water":
            indices = [i for i, farm_plot in enumerate(FARM) if farm_plot["state"] in ("planted", "ready")]
            if indices:
                idx = random.choice(indices)
                FARM[idx]["watered"] += 1
        elif action == "rest":
            rest_day()
        elif action == "harvest":
            for idx, farm_plot in enumerate(FARM):
                if farm_plot["state"] == "ready":
                    harvest_plot()
                    break
        elif action == "sell":
            items = [k for k in INVENTORY.keys() if not k.endswith("_seed")]
            if items:
                item = random.choice(items)
                INVENTORY[item] -= 1
                if INVENTORY[item] <= 0:
                    del INVENTORY[item]

        time.sleep(0.05)


def cheat_menu():
    global MONEY, INVENTORY, FARM, MESSAGE
    print("ЧИТ-МЕНЮ: (1) +$100, (2) заполнить грядки, (3) удалить инвентарь, (4) выйти")
    choice = input("→ ").strip()

    if choice == "1":
        MONEY += 100
        MESSAGE = "Получили $100."
    elif choice == "2":
        for i in range(MAX_PLOTS):
            plant = random.choice(PLANTS)
            FARM[i]["state"] = "planted"
            FARM[i]["plant"] = plant
            FARM[i]["age"] = plant["grow"] - 1
            FARM[i]["watered"] = plant["water_need"]
        MESSAGE = "Грядки заполнены (чит)."
    elif choice == "3":
        INVENTORY = {}
        MESSAGE = "Инвентарь очищен."
    else:
        MESSAGE = "Выход из чита."


def dumb_duplicate_of_stats():
    global DAY, MONEY, INVENTORY, FARM
    total = 0
    ready = 0
    for farm_plot in FARM:
        if farm_plot["state"] in ("planted", "ready"):
            total += 1
        if farm_plot["state"] == "ready":
            ready += 1
    print("День:", DAY)
    print(f"Деньги: ${MONEY:.2f}")
    print("Посажено:", total, "Готово:", ready)
    print("Инвентарь:", INVENTORY)


def startup():
    global SAVE_FILE, MESSAGE
    if os.path.exists(SAVE_FILE):
        try:
            load_game()
        except Exception:
            init_farm()
            MESSAGE = "Ошибка при загрузке. Создали пустую ферму."
    else:
        init_farm()


def quick_test():
    init_farm()
    assert len(FARM) == MAX_PLOTS
    plant = get_plant_by_id(1)
    FARM[0]["state"] = "planted"
    FARM[0]["plant"] = plant
    FARM[0]["age"] = plant["grow"]
    FARM[0]["watered"] = plant["water_need"]
    rest_day()
    return True


if __name__ == "__main__":
    startup()
    print("1 - Играть, 2 - Автоплей (тест), 3 - Тест, 4 - Читы")

    try:
        user_input = input("Выбор: ").strip()
    except Exception:
        user_input = "1"

    if user_input == "2":
        print("Запуск автопилота...")
        autopilot(50)
        print("Автопилот завершён.")
        print("Состояние фермы:")
        show_farm()
        sys.exit(0)
    elif user_input == "3":
        print("Запуск быстрых тестов...")
        ok = quick_test()
        print("Тесты прошли:", ok)
        sys.exit(0)
    elif user_input == "4":
        cheat_menu()

    main_loop()
    print("Пока! Спасибо за игру.")