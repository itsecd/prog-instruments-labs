import pytest
from task_manager import TaskManager


@pytest.fixture
def manager(tmp_path):
    db_file = tmp_path / "temp_tasks.json"
    return TaskManager(filename=str(db_file))


def test_add_task(manager):
    task = manager.add_task("Test Task", "high")
    assert task["id"] == 1
    assert task["title"] == "Test Task"
    assert task["completed"] is False
    assert len(manager.tasks) == 1


def test_complete_task(manager):
    manager.add_task("To be completed")
    completed = manager.complete_task(1)
    assert completed is not None
    assert completed["completed"] is True
    assert len(manager.get_pending_tasks()) == 0


def test_complete_nonexistent_task(manager):
    result = manager.complete_task(999)
    assert result is None
