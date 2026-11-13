"""
Лабораторная работа №5 «Логирование».
Консольный менеджер задач с сохранением данных в JSON.
Вариант проекта: собственное приложение Task Manager.
"""

import json
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "tasks.json"
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "application.log"


def setup_logging():
    """Настраивает вывод логов в консоль и файл с ротацией."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=500_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


logger = logging.getLogger("task_manager")


def load_tasks():
    """Загружает задачи из JSON-файла."""
    logger.info("Начата загрузка задач из %s", DATA_FILE)
    if not DATA_FILE.exists():
        logger.warning("Файл данных отсутствует; будет создан пустой список")
        return []

    try:
        with DATA_FILE.open("r", encoding="utf-8") as file:
            tasks = json.load(file)
        if not isinstance(tasks, list):
            raise ValueError("Корневой элемент JSON должен быть списком")
        logger.info("Загрузка завершена. Загружено задач: %d", len(tasks))
        return tasks
    except (json.JSONDecodeError, OSError, ValueError):
        logger.exception("Не удалось загрузить данные задач")
        raise


def save_tasks(tasks):
    """Сохраняет список задач в JSON-файл."""
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with DATA_FILE.open("w", encoding="utf-8") as file:
            json.dump(tasks, file, ensure_ascii=False, indent=4)
        logger.info("Данные сохранены. Количество задач: %d", len(tasks))
    except OSError:
        logger.exception("Ошибка при сохранении файла задач")
        raise


def get_next_id(tasks):
    """Возвращает следующий свободный идентификатор."""
    if not tasks:
        return 1
    return max(task["id"] for task in tasks) + 1


def show_tasks(tasks):
    """Выводит все задачи."""
    logger.info("Запрошен просмотр списка задач")
    if not tasks:
        print("Список задач пуст.")
        logger.info("Список задач пуст")
        return
    print("\nСписок задач:")
    for task in tasks:
        status = "выполнена" if task["completed"] else "не выполнена"
        print(f'{task["id"]}. {task["title"]} | {status} | {task["created_at"]}')
    logger.info("Показано задач: %d", len(tasks))


def add_task(tasks):
    """Добавляет новую задачу."""
    title = input("Название задачи: ").strip()
    if not title:
        print("Название не может быть пустым.")
        logger.warning("Пользователь попытался добавить задачу без названия")
        return

    description = input("Описание: ").strip()
    task = {
        "id": get_next_id(tasks),
        "title": title,
        "description": description,
        "completed": False,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    tasks.append(task)
    save_tasks(tasks)
    logger.info("Добавлена задача id=%d, title=%r", task["id"], title)
    print(f'Задача "{title}" добавлена.')


def find_task(tasks, task_id):
    """Ищет задачу по идентификатору."""
    for task in tasks:
        if task["id"] == task_id:
            return task
    return None


def complete_task(tasks):
    """Отмечает задачу как выполненную."""
    try:
        task_id = int(input("Введите ID задачи: "))
    except ValueError:
        print("ID должен быть числом.")
        logger.warning("Введён некорректный идентификатор задачи")
        return

    task = find_task(tasks, task_id)
    if task is None:
        print("Задача не найдена.")
        logger.warning("Не найдена задача для завершения: id=%d", task_id)
        return

    if task["completed"]:
        print("Эта задача уже выполнена.")
        logger.info("Повторная попытка завершить задачу id=%d", task_id)
        return

    task["completed"] = True
    save_tasks(tasks)
    logger.info("Задача отмечена выполненной: id=%d", task_id)
    print("Задача отмечена как выполненная.")


def delete_task(tasks):
    """Удаляет задачу."""
    try:
        task_id = int(input("Введите ID задачи для удаления: "))
    except ValueError:
        print("ID должен быть числом.")
        logger.warning("Введён некорректный идентификатор для удаления")
        return

    task = find_task(tasks, task_id)
    if task is None:
        print("Задача не найдена.")
        logger.warning("Попытка удалить несуществующую задачу: id=%d", task_id)
        return

    tasks.remove(task)
    save_tasks(tasks)
    logger.info("Удалена задача id=%d, title=%r", task_id, task["title"])
    print("Задача удалена.")


def search_tasks(tasks):
    """Ищет задачи по части названия."""
    query = input("Введите текст для поиска: ").strip().lower()
    if not query:
        print("Поисковый запрос пуст.")
        logger.warning("Выполнена попытка поиска с пустым запросом")
        return

    found = [task for task in tasks if query in task["title"].lower()]
    logger.info("Поиск по запросу %r. Найдено совпадений: %d", query, len(found))
    if not found:
        print("Совпадений не найдено.")
        return
    show_tasks(found)


def show_statistics(tasks):
    """Показывает краткую статистику."""
    total = len(tasks)
    completed = sum(task["completed"] for task in tasks)
    pending = total - completed
    logger.info(
        "Запрошена статистика: всего=%d, выполнено=%d, осталось=%d",
        total, completed, pending
    )
    print(f"Всего задач: {total}")
    print(f"Выполнено: {completed}")
    print(f"Осталось: {pending}")


def print_menu():
    """Показывает меню приложения."""
    print("\n--- МЕНЕДЖЕР ЗАДАЧ ---")
    print("1. Показать задачи")
    print("2. Добавить задачу")
    print("3. Отметить задачу выполненной")
    print("4. Удалить задачу")
    print("5. Найти задачу")
    print("6. Статистика")
    print("0. Выход")


def main():
    """Главный цикл приложения."""
    logger.info("Приложение запущено")
    tasks = load_tasks()

    while True:
        print_menu()
        choice = input("Выберите действие: ").strip()
        logger.info("Выбрана команда меню: %r", choice)

        if choice == "1":
            show_tasks(tasks)
        elif choice == "2":
            add_task(tasks)
        elif choice == "3":
            complete_task(tasks)
        elif choice == "4":
            delete_task(tasks)
        elif choice == "5":
            search_tasks(tasks)
        elif choice == "6":
            show_statistics(tasks)
        elif choice == "0":
            logger.info("Пользователь завершил работу приложения")
            print("До свидания!")
            break
        else:
            print("Неизвестная команда. Повторите ввод.")
            logger.warning("Получена неизвестная команда меню: %r", choice)

    logger.info("Приложение завершило работу")


if __name__ == "__main__":
    setup_logging()
    try:
        main()
    except Exception:
        logger.exception("Критическая ошибка приложения")
        raise
