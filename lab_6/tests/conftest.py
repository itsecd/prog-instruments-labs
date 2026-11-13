from datetime import date

import pytest

from task_manager import TaskManager


@pytest.fixture
def manager(tmp_path):
    """Provide a manager backed by a temporary JSON file."""
    return TaskManager(tmp_path / "tasks.json")


@pytest.fixture
def sample_manager(manager):
    manager.add_task("Buy groceries", "Milk and bread", priority="high", due_date="2026-11-10")
    manager.add_task("Read a book", "Read two chapters", priority="low", due_date="2026-11-20")
    manager.add_task("Submit report", "Finish the lab report", priority="normal")
    return manager
