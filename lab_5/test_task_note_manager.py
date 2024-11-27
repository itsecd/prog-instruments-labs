import pytest
from task_note_manager import TaskNoteManager


@pytest.fixture
def manager():
    """Создает экземпляр TaskNoteManager для использования в тестах."""
    return TaskNoteManager()


def test_add_task(manager):
    """Проверяет добавление новой задачи и ее корректное сохранение."""
    assert manager.add_task("Task 1") == True
    assert len(manager.get_tasks()) == 1
    assert manager.get_tasks()[0]["task"] == "Task 1"


def test_add_duplicate_task(manager):
    """Проверяет, что повторное добавление одной и той же задачи не сработает."""
    manager.add_task("Task 1")
    assert manager.add_task("Task 1") == False
    assert len(manager.get_tasks()) == 1


def test_remove_task(manager):
    """Проверяет удаление существующей задачи."""
    manager.add_task("Task 1")
    manager.remove_task("Task 1")
    assert len(manager.get_tasks()) == 0


def test_remove_nonexistent_task(manager):
    """Проверяет, что удаление несуществующей задачи не вызывает ошибок."""
    manager.add_task("Task 1")
    manager.remove_task("Nonexistent Task")
    assert len(manager.get_tasks()) == 1


def test_complete_task(manager):
    """Проверяет, что задача может быть отмечена как выполненная."""
    manager.add_task("Task 1")
    assert manager.complete_task("Task 1") == True
    assert manager.get_tasks()[0]["completed"] == True


def test_complete_nonexistent_task(manager):
    """Проверяет, что попытка отметить несуществующую задачу как выполненную вернет False."""
    assert manager.complete_task("Nonexistent Task") == False


def test_incomplete_task(manager):
    """Проверяет, что задача может быть отмечена как невыполненная."""
    manager.add_task("Task 1")
    manager.complete_task("Task 1")
    assert manager.incomplete_task("Task 1") == True
    assert manager.get_tasks()[0]["completed"] == False


def test_get_tasks(manager):
    """Проверяет, что все добавленные задачи могут быть получены."""
    manager.add_task("Task 1")
    manager.add_task("Task 2")
    tasks = manager.get_tasks()
    assert len(tasks) == 2
    assert tasks[0]["task"] == "Task 1"
    assert tasks[1]["task"] == "Task 2"


def test_get_completed_tasks(manager):
    """Проверяет, что можно получить список выполненных задач."""
    manager.add_task("Task 1")
    manager.complete_task("Task 1")
    completed_tasks = manager.get_completed_tasks()
    assert len(completed_tasks) == 1
    assert completed_tasks[0]["task"] == "Task 1"


def test_get_incomplete_tasks(manager):
    """Проверяет, что можно получить список невыполненных задач."""
    manager.add_task("Task 1")
    manager.add_task("Task 2")
    manager.complete_task("Task 1")
    incomplete_tasks = manager.get_incomplete_tasks()
    assert len(incomplete_tasks) == 1
    assert incomplete_tasks[0]["task"] == "Task 2"


def test_get_tasks_by_tag(manager):
    """Проверяет, что задачи могут быть отфильтрованы по тегам."""
    manager.add_task("Task 1", tags=["work"])
    manager.add_task("Task 2", tags=["personal"])
    tasks = manager.get_tasks_by_tag("work")
    assert len(tasks) == 1
    assert tasks[0]["task"] == "Task 1"


def test_add_note(manager):
    """Проверяет добавление новой заметки и ее корректное сохранение."""
    assert manager.add_note("This is a note") == True
    assert len(manager.get_notes()) == 1
    assert manager.get_notes()[0] == "This is a note"


def test_add_empty_note(manager):
    """Проверяет, что добавление пустой заметки не сработает."""
    assert manager.add_note("") == False
    assert len(manager.get_notes()) == 0


def test_remove_note(manager):
    """Проверяет удаление существующей заметки."""
    manager.add_note("This is a note")
    manager.remove_note("This is a note")
    assert len(manager.get_notes()) == 0


def test_remove_nonexistent_note(manager):
    """Проверяет, что удаление несуществующей заметки не вызывает ошибок."""
    manager.add_note("This is a note")
    manager.remove_note("Nonexistent Note")
    assert len(manager.get_notes()) == 1


def test_get_notes(manager):
    """Проверяет, что все добавленные заметки могут быть получены."""
    manager.add_note("Note 1")
    manager.add_note("Note 2")
    notes = manager.get_notes()
    assert len(notes) == 2
    assert notes[0] == "Note 1"
    assert notes[1] == "Note 2"


def test_add_task_with_tags(manager):
    """Проверяет добавление задачи с несколькими тегами."""
    assert manager.add_task("Task 1", tags=["work", "urgent"]) == True
    tasks = manager.get_tasks()
    assert len(tasks) == 1
    assert tasks[0]["task"] == "Task 1"
    assert set(tasks[0]["tags"]) == {"work", "urgent"}


def test_get_tasks_by_multiple_tags(manager):
    """Проверяет фильтрацию задач по нескольким тегам."""
    manager.add_task("Task 1", tags=["work", "urgent"])
    manager.add_task("Task 2", tags=["personal"])
    manager.add_task("Task 3", tags=["work"])

    tasks = manager.get_tasks_by_tag("work")
    assert len(tasks) == 2  # Task 1 and Task 3
    assert all(task["task"] in ["Task 1", "Task 3"] for task in tasks)


def test_remove_task_with_notes(manager):
    """Проверяет, что задача с заметками может быть удалена корректно."""
    manager.add_task("Task 1")
    manager.add_note("Note for Task 1")
    manager.remove_task("Task 1")
    assert len(manager.get_tasks()) == 0
    assert len(manager.get_notes()) == 1


def test_large_number_of_tasks(manager):
    """Проверяет производительность при добавлении большого количества задач."""
    for i in range(1000):
        manager.add_task(f"Task {i}")
    assert len(manager.get_tasks()) == 1000


def test_unique_tags(manager):
    """Проверяет, что теги задач уникальны."""
    manager.add_task("Task 1", tags=["work"])
    manager.add_task("Task 2", tags=["work"])
    tasks = manager.get_tasks_by_tag("work")
    assert len(tasks) == 2


def test_note_with_special_characters(manager):
    """Проверяет, что заметки могут содержать специальные символы."""
    assert manager.add_note("Note with special characters: !@#$%^&*()") == True
    assert len(manager.get_notes()) == 1
    assert manager.get_notes()[0] == "Note with special characters: !@#$%^&*()"


def test_task_and_note_same_name(manager):
    """Проверяет, что задачи и заметки могут иметь одинаковые названия."""
    manager.add_task("Same Name")
    assert manager.add_note("Same Name") == True
    assert len(manager.get_notes()) == 1
    assert len(manager.get_tasks()) == 1


@pytest.mark.parametrize("task_name, expected_tags", [
    ("Task 1", ["work", "urgent"]),
    ("Task 2", ["personal"]),
    ("Task 3", ["work", "home"]),
])
def test_add_task_with_tags(manager, task_name, expected_tags):
    """Проверяет добавление задач с различными тегами."""
    assert manager.add_task(task_name, tags=expected_tags) == True
    tasks = manager.get_tasks()
    assert len(tasks) == 1
    assert tasks[0]["task"] == task_name
    assert set(tasks[0]["tags"]) == set(expected_tags)


@pytest.mark.parametrize("task_name, note_name", [
    ("Task A", "Note A"),
    ("Task B", "Note B"),
    ("Task C", "Note C"),
])
def test_task_and_note_same_name(manager, task_name, note_name):
    """Проверяет, что задачи и заметки могут иметь одинаковые названия."""
    manager.add_task(task_name)
    assert manager.add_note(note_name) == True
    assert len(manager.get_notes()) == 1
    assert len(manager.get_tasks()) == 1
    assert manager.get_notes()[0] == note_name
    assert manager.get_tasks()[0]["task"] == task_name


@pytest.mark.parametrize("task_name, expected_completed", [
    ("Task 1", True),
    ("Task 2", False),
])
def test_complete_and_incomplete_task(manager, task_name, expected_completed):
    """Проверяет, что задача может быть отмечена как выполненная и невыполненная."""
    manager.add_task(task_name)
    if expected_completed:
        manager.complete_task(task_name)
    else:
        manager.complete_task(task_name)
        manager.incomplete_task(task_name)

    assert manager.get_tasks()[0]["completed"] == expected_completed


@pytest.mark.parametrize("note_content", [
    "Note with special characters: !@#$%^&*()",
    "Another note with spaces and symbols: %^&*()",
    "Short note",
])
def test_add_notes_with_various_content(manager, note_content):
    """Проверяет добавление заметок с различным содержимым."""
    assert manager.add_note(note_content) == True
    assert len(manager.get_notes()) == 1
    assert manager.get_notes()[0] == note_content
