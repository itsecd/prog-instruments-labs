"""
Консольный менеджер задач.
Исходная версия проекта до добавления логирования.
Позволяет создавать, просматривать, изменять, удалять и искать задачи.
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "tasks.json"


def load_tasks():
    """Загружает задачи из JSON-файла."""
    if not DATA_FILE.exists():
        return []
    try:
        with DATA_FILE.open("r", encoding="utf-8") as file:
            tasks = json.load(file)
        if not isinstance(tasks, list):
            raise ValueError("Файл должен содержать список задач.")
        return tasks
    except (json.JSONDecodeError, OSError) as error:
        print(f"Не удалось прочитать файл: {error}")
        return []


def save_tasks(tasks):
    """Сохраняет список задач в JSON-файл."""
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with DATA_FILE.open("w", encoding="utf-8") as file:
        json.dump(tasks, file, ensure_ascii=False, indent=4)


def get_next_id(tasks):
    """Возвращает следующий свободный идентификатор."""
    if not tasks:
        return 1
    return max(task["id"] for task in tasks) + 1


def show_tasks(tasks):
    """Выводит все задачи."""
    if not tasks:
        print("Список задач пуст.")
        return
    print("\nСписок задач:")
    for task in tasks:
        status = "выполнена" if task["completed"] else "не выполнена"
        print(f'{task["id"]}. {task["title"]} | {status} | {task["created_at"]}')


def add_task(tasks):
    """Добавляет новую задачу."""
    title = input("Название задачи: ").strip()
    if not title:
        print("Название не может быть пустым.")
        return
    description = input("Описание: ").strip()
    task = {
        "id": get_next_id(tasks),
        "title": title,
        "description": description,
        "completed": False,
        "created_at": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    tasks.append(task)
    save_tasks(tasks)
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
        return
    task = find_task(tasks, task_id)
    if task is None:
        print("Задача не найдена.")
        return
    task["completed"] = True
    save_tasks(tasks)
    print("Задача отмечена как выполненная.")


def delete_task(tasks):
    """Удаляет задачу."""
    try:
        task_id = int(input("Введите ID задачи для удаления: "))
    except ValueError:
        print("ID должен быть числом.")
        return
    task = find_task(tasks, task_id)
    if task is None:
        print("Задача не найдена.")
        return
    tasks.remove(task)
    save_tasks(tasks)
    print("Задача удалена.")


def search_tasks(tasks):
    """Ищет задачи по части названия."""
    query = input("Введите текст для поиска: ").strip().lower()
    if not query:
        print("Поисковый запрос пуст.")
        return
    found = [task for task in tasks if query in task["title"].lower()]
    if not found:
        print("Совпадений не найдено.")
        return
    show_tasks(found)


def show_statistics(tasks):
    """Показывает краткую статистику."""
    total = len(tasks)
    completed = sum(task["completed"] for task in tasks)
    pending = total - completed
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
    tasks = load_tasks()
    while True:
        print_menu()
        choice = input("Выберите действие: ").strip()
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
            print("До свидания!")
            break
        else:
            print("Неизвестная команда. Повторите ввод.")


if __name__ == "__main__":
    main()
