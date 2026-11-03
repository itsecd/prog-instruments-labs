from datetime import datetime
import json
import os
from typing import Any


class TaskManager:
    def __init__(self, filename: str = "tasks.json") -> None:
        self.filename: str = filename
        self.tasks: list[dict[str, Any]] = []
        self.load_tasks()

    def add_task(self, title: str, priority: str = "medium") -> dict[str, Any]:
        new_task: dict[str, Any] = {
            "id": len(self.tasks) + 1,
            "title": title,
            "priority": priority,
            "completed": False,
            "created_at": datetime.now().isoformat(),
        }
        self.tasks.append(new_task)
        return new_task

    def complete_task(self, task_id: int) -> dict[str, Any] | None:
        for task in self.tasks:
            if task["id"] == task_id:
                task["completed"] = True
                return task
        return None

    def get_pending_tasks(self) -> list[dict[str, Any]]:
        return [t for t in self.tasks if not t["completed"]]

    def save_tasks(self) -> None:
        with open(self.filename, "w", encoding="utf-8") as f:
            json.dump(self.tasks, f, indent=4)

    def load_tasks(self) -> None:
        if os.path.exists(self.filename):
            try:
                with open(self.filename, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.tasks = data
            except Exception:
                self.tasks = []


def main() -> None:
    manager = TaskManager(filename="test_tasks.json")
    t1 = manager.add_task("Настроить Ruff", "high")
    manager.add_task("Настроить Pyright", "medium")
    print(f"Добавлена задача: {t1['title']}")
    manager.complete_task(1)
    print(f"Осталось задач: {len(manager.get_pending_tasks())}")


if __name__ == "__main__":
    main()