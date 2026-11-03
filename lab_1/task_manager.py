import os, sys
import json
from datetime import datetime

class TaskManager:
    def __init__(self, filename="tasks.json"):
        self.filename = filename
        self.tasks = []
        self.load_tasks()

    def add_task(self, title, priority = "medium"):
        new_task = {
            'id': len(self.tasks) + 1,
            'title': title,
            'priority': priority,
            'completed': False,
            'created_at': datetime.now().isoformat()
        }
        self.tasks.append(new_task)
        return new_task

    def complete_task(self, task_id):
        for task in self.tasks:
            if task['id'] == task_id:
                task['completed'] = True
                return task
        return None

    def get_pending_tasks(self):
        return [t for t in self.tasks if not t['completed']]

    def save_tasks(self):
        with open(self.filename, 'w', encoding='utf-8') as f:
            json.dump(self.tasks, f, indent=4)

    def load_tasks(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r', encoding='utf-8') as f:
                    self.tasks = json.load(f)
            except Exception:
                self.tasks = []

def main():
    manager = TaskManager(filename="test_tasks.json")
    t1 = manager.add_task("Настроить Ruff", "high")
    t2 = manager.add_task("Настроить Pyright", "medium")
    print(f"Добавлена задача: {t1['title']}")
    manager.complete_task(1)
    print(f"Осталось задач: {len(manager.get_pending_tasks())}")

if __name__ == "__main__":
    main()