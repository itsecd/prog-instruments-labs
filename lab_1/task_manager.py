import datetime


class TaskManager:
    def __init__(self):
        self.tasks = []
        self._next_id = 1

    def add_task(self, title, priority="normal"):
        if not title:
            print("Ошибка: Название задачи не может быть пустым")
            return None

        task = {
            "id": self._next_id,
            "title": title,
            "priority": priority,
            "completed": False,
            "created_at": str(datetime.datetime.now()),
        }
        self.tasks.append(task)
        self._next_id += 1
        return task

    def mark_completed(self, task_id):
        for task in self.tasks:
            if task["id"] == task_id:
                task["completed"] = True
                return task
        return None

    def get_pending_tasks(self):
        pending = []
        for task in self.tasks:
            if not task["completed"]:
                pending.append(task)
        return pending

    def delete_task(self, task_id):
        initial_length = len(self.tasks)
        self.tasks = [t for t in self.tasks if t["id"] != task_id]
        return len(self.tasks) < initial_length


def format_task(task):
    if task is None:
        return "Задача не найдена"
    status = "+" if task["completed"] else " "
    return "[" + status + "] #" + str(task["id"]) + " " + task["title"] + " (" + task["priority"] + ")"


if __name__ == "__main__":
    manager = TaskManager()

    t1 = manager.add_task("Купить хлеб", priority="high")
    t2 = manager.add_task("Сделать лабораторную")
    manager.add_task("")

    print(format_task(t1))
    print(format_task(t2))

    manager.mark_completed(1)
    print(format_task(t1))
    print("\nНевыполненные задачи:")
    for t in manager.get_pending_tasks():
        print(format_task(t))

    manager.delete_task(2)
    print("\nОсталось задач:", len(manager.tasks))