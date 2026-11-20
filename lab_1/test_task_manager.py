from lab_1.task_manager import TaskManager, format_task


def test_add_task() -> None:
    manager = TaskManager()
    task = manager.add_task("Тестовая задача", priority="high")
    assert task is not None
    assert task["id"] == 1
    assert task["title"] == "Тестовая задача"
    assert task["completed"] is False


def test_add_empty_task() -> None:
    manager = TaskManager()
    task = manager.add_task("")
    assert task is None


def test_mark_completed() -> None:
    manager = TaskManager()
    task = manager.add_task("Задача")
    assert task is not None
    updated = manager.mark_completed(task["id"])
    assert updated is not None
    assert updated["completed"] is True


def test_delete_task() -> None:
    manager = TaskManager()
    task = manager.add_task("Удалить меня")
    assert task is not None
    assert manager.delete_task(task["id"]) is True
    assert len(manager.tasks) == 0


def test_format_task_none() -> None:
    assert format_task(None) == "Задача не найдена"
