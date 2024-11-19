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
