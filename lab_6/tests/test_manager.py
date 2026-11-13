import json
from datetime import date
from unittest.mock import patch

import pytest

from task_manager import (
    StorageError,
    TaskManager,
    TaskNotFoundError,
    ValidationError,
)


def test_new_manager_starts_with_empty_list(manager):
    assert manager.tasks == []


def test_add_task_creates_task_and_saves_it(manager, tmp_path):
    task = manager.add_task("  Prepare presentation  ", "Slides", priority="high")

    assert task.task_id == 1
    assert task.title == "Prepare presentation"
    assert task.description == "Slides"
    assert (tmp_path / "tasks.json").exists()


@pytest.mark.parametrize("title", ["", "   ", "x" * 101])
def test_add_task_rejects_invalid_title(manager, title):
    with pytest.raises(ValidationError):
        manager.add_task(title)


@pytest.mark.parametrize("priority", ["urgent", "HIGH", "", "medium"])
def test_add_task_rejects_unknown_priority(manager, priority):
    with pytest.raises(ValidationError):
        manager.add_task("Valid title", priority=priority)


@pytest.mark.parametrize("due_date", ["2026/11/13", "31-12-2026", "2026-02-30"])
def test_add_task_rejects_invalid_due_date(manager, due_date):
    with pytest.raises(ValidationError):
        manager.add_task("Valid title", due_date=due_date)


def test_tasks_are_restored_after_creating_new_manager(manager, tmp_path):
    manager.add_task("Persist me")

    restored = TaskManager(tmp_path / "tasks.json")

    assert len(restored.tasks) == 1
    assert restored.get_task(1).title == "Persist me"


def test_update_task_changes_only_requested_fields(sample_manager):
    updated = sample_manager.update_task(1, title="Buy food", priority="normal")

    assert updated.title == "Buy food"
    assert updated.priority == "normal"
    assert updated.description == "Milk and bread"


def test_update_task_rejects_unknown_field(sample_manager):
    with pytest.raises(ValidationError, match="Unsupported fields"):
        sample_manager.update_task(1, owner="Alex")


def test_complete_task_changes_status(sample_manager):
    completed = sample_manager.complete_task(2)

    assert completed.completed is True
    assert sample_manager.list_tasks(completed=True) == [completed]


def test_delete_task_removes_task(sample_manager):
    sample_manager.delete_task(2)

    assert [task.task_id for task in sample_manager.tasks] == [1, 3]


def test_get_unknown_task_raises_error(manager):
    with pytest.raises(TaskNotFoundError):
        manager.get_task(404)


def test_list_tasks_filters_by_priority_and_status(sample_manager):
    sample_manager.complete_task(1)

    result = sample_manager.list_tasks(completed=False, priority="low")

    assert [task.title for task in result] == ["Read a book"]


def test_search_is_case_insensitive_and_checks_description(sample_manager):
    result = sample_manager.search_tasks("LAB")

    assert [task.title for task in result] == ["Submit report"]


def test_overdue_tasks_excludes_completed_and_future_tasks(sample_manager):
    sample_manager.complete_task(1)

    result = sample_manager.overdue_tasks(today=date(2026, 11, 13))

    assert result == []


def test_save_uses_json_serializer(manager):
    manager.add_task("A task")

    with patch("task_manager.manager.json.dump", wraps=json.dump) as json_dump:
        manager.save()

    json_dump.assert_called_once()
    assert json_dump.call_args.kwargs["ensure_ascii"] is False


def test_corrupted_json_raises_storage_error(tmp_path):
    storage_file = tmp_path / "broken.json"
    storage_file.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(StorageError, match="Could not load tasks"):
        TaskManager(storage_file)


def test_write_failure_is_reported_as_storage_error(manager):
    with patch("pathlib.Path.open", side_effect=OSError("disk unavailable")):
        with pytest.raises(StorageError, match="Could not save tasks"):
            manager.save()
