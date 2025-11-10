import random
import time
import json
import os
import sys


MONEY = 100.0
DAY = 1
INVENT = {}
FARM = []
WEATHER = "sunny"
PLAYER_NAME = "Farmer"
GAME_OVER = False
MSG = ""
SAVE_FILE = "bad_farm_save.json"
PLANTS = [
    {"id": 1, "name": "Carrot", "grow": 3, "water_need": 1, "price": 2.5},
    {"id": 2, "name": "Potato", "grow": 5, "water_need": 2, "price": 4.0},
    {"id": 3, "name": "Tomato", "grow": 4, "water_need": 2, "price": 5.0},
    {"id": 4, "name": "Corn", "grow": 7, "water_need": 3, "price": 7.5},
    {"id": 5, "name": "Pumpkin", "grow": 10, "water_need": 4, "price": 15.0},
]
MAX_PLOTS = 6
x = 0
y = 1
z = 2


def get_plant_by_id(pid):
    for p in PLANTS:
        if p["id"] == pid:
            return p
    return None


def init_farm():
    global FARM, INVENT, MONEY, DAY, PLAYER_NAME
    FARM = []
    INVENT = {}
    for i in range(MAX_PLOTS):
        FARM.append({"plot": i, "state": "empty", "plant": None, "age": 0, "watered": 0})
    MONEY = 100.0
    DAY = 1
    PLAYER_NAME = "Farmer"


def get_plot_display_string(plot_data):
    """Формирует строку для отображения состояния грядки"""
    plot_num = plot_data["plot"]
    state = plot_data["state"]

    display_string = "[{}]".format(plot_num)

    if state == "empty":
        display_string += " Пусто"
    elif state == "planted":
        p = plot_data["plant"]
        display_string += " {} ({}/{}) вода:{}/{}".format(
            p["name"], plot_data["age"], p["grow"],
            plot_data["watered"], p["water_need"]
        )
    elif state == "ready":
        p = plot_data["plant"]
        display_string += " {} (Готов)".format(p["name"])
    elif state == "withered":
        display_string += " Завяли"
    else:
        display_string += " ???"

    return display_string


def show_farm():
    global FARM, DAY, MONEY, WEATHER, MSG, INVENT
    print("\n" + "=" * 40)
    print("День: {} | Деньги: ${:.2f} | Погода: {}".format(DAY, MONEY, WEATHER))
    print("Инвентарь:", INVENT)
    print("-" * 40)

    for plot in FARM:
        print(get_plot_display_string(plot))

    print("-" * 40)
    if MSG:
        print(">>>", MSG)
    print("=" * 40)


def get_valid_input(prompt, input_type=int, validation_func=None):
    """Универсальная функция для получения и валидации ввода"""
    try:
        value = input_type(input(prompt))
        if validation_func and not validation_func(value):
            return None
        return value
    except (ValueError, TypeError):
        return None


def plant_seed():
    global FARM, MSG, MONEY

    pid = get_valid_input(
        "Введите ID растения для посадки (1-5): ",
        int,
        lambda x: 1 <= x <= 5
    )
    if pid is None:
        MSG = "Неправильный ID."
        return

    p = get_plant_by_id(pid)
    if not p:
        MSG = "Неправильный ID."
        return

    plot = get_valid_input(
        "В какую грядку (0-{}): ".format(MAX_PLOTS - 1),
        int,
        lambda x: 0 <= x < MAX_PLOTS
    )
    if plot is None:
        MSG = "Неправильная грядка."
        return

    if FARM[plot]["state"] != "empty":
        MSG = "Грядка уже занята."
        return


def water_plot():
    global FARM, MSG
    try:
        plot = int(input("Какую грядку полить (0-{}): ".format(MAX_PLOTS - 1)))
    except:
        plot = -1
    if plot < 0 or plot >= MAX_PLOTS:
        MSG = "Неправильная грядка."
        return
    if FARM[plot]["state"] not in ("planted", "ready"):
        MSG = "Там нечего поливать."
        return
    FARM[plot]["watered"] += 1
    MSG = "Полили грядку {}.".format(plot)


def harvest_plot():
    global FARM, MONEY, INVENT, MSG
    try:
        plot = int(input("Какую грядку собрать (0-{}): ".format(MAX_PLOTS - 1)))
    except:
        plot = -1
    if plot < 0 or plot >= MAX_PLOTS:
        MSG = "Неправильная грядка."
        return
    if FARM[plot]["state"] != "ready":
        MSG = "Там ещё не готово."
        return
    p = FARM[plot]["plant"]
    amount = random.randint(1, 3) + int(p["grow"] / 4)
    name = p["name"]
    INVENT[name] = INVENT.get(name, 0) + amount
    FARM[plot]["state"] = "empty"
    FARM[plot]["plant"] = None
    FARM[plot]["age"] = 0
    FARM[plot]["watered"] = 0
    MSG = "Собрали {} x{}.".format(name, amount)
    if random.random() < 0.05:
        MSG += " Лопата сломалась (ничего не делает)."


def sell_items():
    global INVENT, MONEY, MSG
    print("Инвентарь:", INVENT)
    item = input("Что продать (название) или оставить пустым: ").strip()
    if item == "":
        MSG = "Ничего не продали."
        return
    if item not in INVENT or INVENT[item] <= 0:
        MSG = "У вас нет такого товара."
        return
    try:
        cnt = int(input("Сколько продать (количество): "))
    except:
        cnt = 0
    if cnt <= 0 or cnt > INVENT[item]:
        MSG = "Неправильное количество."
        return
    base_price = 0.0
    for p in PLANTS:
        if p["name"] == item:
            base_price = p["price"]
            break
    price = base_price * (0.8 + (DAY % 5) * 0.05)
    MONEY += price * cnt
    INVENT[item] -= cnt
    if INVENT[item] == 0:
        del INVENT[item]
    MSG = "Продали {} x{} за ${:.2f}.".format(item, cnt, price * cnt)


def rest_day():
    global DAY, FARM, WEATHER, MSG
    DAY += 1
    r = random.random()
    if r < 0.6:
        WEATHER = "sunny"
    elif r < 0.85:
        WEATHER = "rain"
    else:
        WEATHER = "storm"
    for f in FARM:
        try:
            if f["state"] == "planted":
                if WEATHER == "rain":
                    f["watered"] += 2
                if WEATHER == "storm":
                    if random.random() < 0.2:
                        f["state"] = "withered"
                        f["plant"] = None
                        f["age"] = 0
                        f["watered"] = 0
                        continue
                f["age"] += 1
                p = f["plant"]
                if f["age"] >= p["grow"] and f["watered"] >= p["water_need"]:
                    f["state"] = "ready"
                elif f["age"] >= p["grow"] and f["watered"] < p["water_need"]:
                    f["state"] = "withered"
                if WEATHER == "sunny":
                    f["watered"] = max(0, f["watered"] - 1)
            elif f["state"] == "withered":
                if random.random() < 0.05:
                    f["state"] = "empty"
                    f["plant"] = None
            else:
                pass
        except Exception:
            pass
    MSG = "Прошёл день."


# helper function to calculate seed cost
def calculate_seed_cost(plant, use_quick_calculation=True):
    if use_quick_calculation:
        return plant["price"] * 0.5 + plant["grow"] * 0.1
    else:
        return plant["price"] * 0.5


# unified seed purchase function
def buy_seeds(use_quick_calculation=True):
    global MONEY, MSG
    if use_quick_calculation:
        print("Семена в магазине:")
        for p in PLANTS:
            cost = calculate_seed_cost(p, True)
            print(p["id"], p["name"], "цена семян ~", cost)
    else:
        print("Добро пожаловать в магазин семян.")
        for p in PLANTS:
            cost = calculate_seed_cost(p, False)
            print("{} - {} - ${:.2f}".format(p["id"], p["name"], cost))

    try:
        pid = int(input("ID семян: " if use_quick_calculation else "Введите ID: "))
    except:
        pid = -1

    p = get_plant_by_id(pid)
    if not p:
        MSG = "Не выбран тип семян." if use_quick_calculation else "Неверный выбор."
        return

    try:
        cnt = int(input("Сколько семян купить: " if use_quick_calculation else "Кол-во: "))
    except:
        cnt = 0

    cost = calculate_seed_cost(p, use_quick_calculation) * cnt

    if cnt <= 0:
        MSG = "Ничего не купили."
        return

    if MONEY < cost:
        MSG = "Недостаточно денег." if use_quick_calculation else "Не хватает денег."
        return

    MONEY -= cost
    name = p["name"]
    INVENT[name + "_seed"] = INVENT.get(name + "_seed", 0) + cnt
    MSG = "Купили {} x{} семян.".format(name, cnt)


# quick purchase menu
def buy_seeds_quick():
    buy_seeds(use_quick_calculation=True)


# shop purchase function
def buy_seeds_shop():
    buy_seeds(use_quick_calculation=False)


def plant_from_invent():
    global INVENT, FARM, MSG
    print("Ваши семена:", {k: v for k, v in INVENT.items() if k.endswith("_seed")})
    s = input("Введите название семяни (например Carrot) или пусто: ").strip()
    if s == "":
        MSG = "Отменено."
        return
    seed_key = s + "_seed"
    if INVENT.get(seed_key, 0) <= 0:
        MSG = "Нет таких семян."
        return
    try:
        plot = int(input("Грядка (0-{}): ".format(MAX_PLOTS - 1)))
    except:
        plot = -1
    if plot < 0 or plot >= MAX_PLOTS:
        MSG = "Неправильная грядка."
        return
    if FARM[plot]["state"] != "empty":
        MSG = "Грядка занята."
        return
    p = None
    for pl in PLANTS:
        if pl["name"].lower() == s.lower():
            p = pl
            break
    if not p:
        MSG = "Растение неизвестно."
        return
    INVENT[seed_key] -= 1
    if INVENT[seed_key] == 0:
        del INVENT[seed_key]
    FARM[plot]["state"] = "planted"
    FARM[plot]["plant"] = p
    FARM[plot]["age"] = 0
    FARM[plot]["watered"] = 0
    MSG = "Посадили семя {}.".format(p["name"])


def save_game():
    global SAVE_FILE, DAY, MONEY, INVENT, FARM, PLAYER_NAME, WEATHER, MSG
    data = {
        "day": DAY,
        "money": MONEY,
        "invent": INVENT,
        "farm": FARM,
        "player": PLAYER_NAME,
        "weather": WEATHER,
    }
    try:
        with open(SAVE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
        MSG = "Игра сохранена."
    except Exception as e:
        MSG = "Ошибка сохранения: {}".format(e)


def load_game():
    global SAVE_FILE, DAY, MONEY, INVENT, FARM, PLAYER_NAME, WEATHER, MSG
    if not os.path.exists(SAVE_FILE):
        MSG = "Файл сохранения не найден."
        return
    try:
        with open(SAVE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        DAY = data.get("day", 1)
        MONEY = data.get("money", 100.0)
        INVENT = data.get("invent", {})
        FARM = data.get("farm", FARM)
        PLAYER_NAME = data.get("player", PLAYER_NAME)
        WEATHER = data.get("weather", "sunny")
        MSG = "Игра загружена."
    except Exception:
        MSG = "Ошибка загрузки."


def random_events():
    global MONEY, INVENT, MSG
    r = random.random()
    if r < 0.1:
        lost = int(MONEY * 0.1)
        MONEY -= lost
        MSG = "Ночной вор унес ${}.".format(lost)
    elif r < 0.2:
        pl = random.choice(PLANTS)
        INVENT[pl["name"] + "_seed"] = INVENT.get(pl["name"] + "_seed", 0) + 1
        MSG = "Нашли семя {} на дороге.".format(pl["name"])
    elif r < 0.25:
        bonus = 10
        MONEY += bonus
        MSG = "Посетили ярмарку, получили бонус ${}.".format(bonus)
    else:
        MSG = "Тихий день."


def market_prices():
    global PLANTS, DAY
    print("Рыночные цены (меняются):")
    for p in PLANTS:
        pr = p["price"] * (0.9 + (DAY % 7) * 0.02)
        print(p["name"], "${:.2f}".format(pr))


def stats():
    global DAY, MONEY, INVENT, FARM
    total_planted = sum(1 for f in FARM if f["state"] in ("planted", "ready"))
    total_ready = sum(1 for f in FARM if f["state"] == "ready")
    print("Статистика --- День {}. Деньги: ${:.2f}".format(DAY, MONEY))
    print("Посажено:", total_planted, "Грядок готово:", total_ready, "Инвентарь:", INVENT)


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


def handle_choice(c):
    global GAME_OVER, MSG
    if c == "1":
        show_farm()
    elif c == "2":
        plant_seed()
    elif c == "3":
        buy_seeds_quick()
    elif c == "4":
        plant_from_invent()
    elif c == "5":
        water_plot()
    elif c == "6":
        harvest_plot()
    elif c == "7":
        sell_items()
    elif c == "8":
        rest_day()
        random_events()
    elif c == "9":
        save_game()
    elif c == "10":
        load_game()
    elif c == "11":
        market_prices()
    elif c == "12":
        stats()
    elif c == "13":
        GAME_OVER = True
    elif c == "h":
        help_menu()
    else:
        MSG = "Неизвестная команда."


def main_loop():
    global GAME_OVER, MSG, PLAYER_NAME, MONEY
    print("Добро пожаловать на плохую ферму!")
    try:
        nm = input("Введите имя фермера (Enter для '{}' ): ".format(PLAYER_NAME)).strip()
        if nm != "":
            PLAYER_NAME = nm
    except:
        pass
    MSG = "Начинаем!"
    count = 0
    while not GAME_OVER:
        if count % 5 == 0:
            r = random.random()
            if r < 0.5:
                pass
            else:
                pass
        show_farm()
        print("Команды: 1..13, h - помощь")
        ch = input("Ваш выбор: ").strip()
        handle_choice(ch)
        if MONEY < 5 and INVENT:
            try:
                it = next(iter(INVENT))
                if it.endswith("_seed"):
                    pass
                else:
                    sold = 1
                    MONEY += sold * (next((p["price"] for p in PLANTS if p["name"] == it), 1.0))
                    INVENT[it] -= sold
                    if INVENT[it] <= 0:
                        del INVENT[it]
                    MSG = "Автопродажа: {} x{}.".format(it, sold)
            except Exception:
                pass
        count += 1
        try:
            time.sleep(0.2)
        except:
            pass


def autopilot(turns=20):
    global MESSAGE, MSG
    for i in range(turns):
        a = random.choice(["plant", "buy", "water", "rest", "harvest", "sell"])
        if a == "plant":
            for idx, f in enumerate(FARM):
                if f["state"] == "empty":
                    p = random.choice(PLANTS)
                    FARM[idx]["state"] = "planted"
                    FARM[idx]["plant"] = p
                    FARM[idx]["age"] = 0
                    FARM[idx]["watered"] = 0
                    break
        elif a == "buy":
            pl = random.choice(PLANTS)
            INVENT[pl["name"] + "_seed"] = INVENT.get(pl["name"] + "_seed", 0) + 1
        elif a == "water":
            idxs = [i for i, f in enumerate(FARM) if f["state"] in ("planted", "ready")]
            if idxs:
                idx = random.choice(idxs)
                FARM[idx]["watered"] += 1
        elif a == "rest":
            rest_day()
        elif a == "harvest":
            for idx, f in enumerate(FARM):
                if f["state"] == "ready":
                    harvest_plot()
                    break
        elif a == "sell":
            items = [k for k in INVENT.keys() if not k.endswith("_seed")]
            if items:
                it = random.choice(items)
                INVENT[it] -= 1
                if INVENT[it] <= 0:
                    del INVENT[it]
        time.sleep(0.05)


def cheat_menu():
    global MONEY, INVENT, FARM, MSG
    print("ЧИТ-МЕНЮ: (1) +$100, (2) заполнить грядки, (3) удалить инвентарь, (4) выйти")
    ch = input("→ ").strip()
    if ch == "1":
        MONEY += 100
        MSG = "Получили $100."
    elif ch == "2":
        for i in range(MAX_PLOTS):
            p = random.choice(PLANTS)
            FARM[i]["state"] = "planted"
            FARM[i]["plant"] = p
            FARM[i]["age"] = p["grow"] - 1
            FARM[i]["watered"] = p["water_need"]
        MSG = "Грядки заполнены (чит)."
    elif ch == "3":
        INVENT = {}
        MSG = "Инвентарь очищен."
    else:
        MSG = "Выход из чита."


def dumb_duplicate_of_stats():
    global DAY, MONEY, INVENT, FARM
    total = 0
    ready = 0
    for f in FARM:
        if f["state"] in ("planted", "ready"):
            total += 1
        if f["state"] == "ready":
            ready += 1
    print("День:", DAY)
    print("Деньги: ${:.2f}".format(MONEY))
    print("Посажено:", total, "Готово:", ready)
    print("Инвентарь:", INVENT)


def startup():
    global SAVE_FILE, MSG
    if os.path.exists(SAVE_FILE):
        try:
            load_game()
        except:
            init_farm()
            MSG = "Ошибка при загрузке. Создали пустую ферму."
    else:
        init_farm()


def quick_test():
    init_farm()
    assert len(FARM) == MAX_PLOTS
    p = get_plant_by_id(1)
    FARM[0]["state"] = "planted"
    FARM[0]["plant"] = p
    FARM[0]["age"] = p["grow"]
    FARM[0]["watered"] = p["water_need"]
    rest_day()
    return True


if __name__ == "__main__":
    startup()
    print("1 - Играть, 2 - Автоплей (тест), 3 - Тест, 4 - Читы")
    try:
        s = input("Выбор: ").strip()
    except:
        s = "1"
    if s == "2":
        print("Запуск автопилота...")
        autopilot(50)
        print("Автопилот завершён.")
        print("Состояние фермы:")
        show_farm()
        sys.exit(0)
    elif s == "3":
        print("Запуск быстрых тестов...")
        ok = quick_test()
        print("Тесты прошли:", ok)
        sys.exit(0)
    elif s == "4":
        cheat_menu()
    main_loop()
    print("Пока! Спасибо за игру.")