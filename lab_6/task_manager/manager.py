"""A small task-management application with JSON persistence.

The module contains validation, CRUD operations, filtering and persistence.
It is intentionally independent from the command-line interface so that its
business logic can be tested with pytest.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any


class TaskManagerError(Exception):
    """Base exception for task manager errors."""


class ValidationError(TaskManagerError):
    """Raised when task data does not satisfy validation rules."""


class TaskNotFoundError(TaskManagerError):
    """Raised when an operation references an unknown task ID."""


class StorageError(TaskManagerError):
    """Raised when task data cannot be read from or written to disk."""


@dataclass
class Task:
    """A task stored by the manager."""

    task_id: int
    title: str
    description: str = ""
    priority: str = "normal"
    due_date: str | None = None
    completed: bool = False


class TaskManager:
    """Manage tasks and persist them in a JSON file."""

    ALLOWED_PRIORITIES = {"low", "normal", "high"}

    def __init__(self, storage_path: str | Path) -> None:
        self.storage_path = Path(storage_path)
        self.tasks: list[Task] = []
        self._load()

    @staticmethod
    def _validate_title(title: str) -> str:
        if not isinstance(title, str):
            raise ValidationError("Task title must be a string.")
        cleaned = title.strip()
        if not cleaned:
            raise ValidationError("Task title cannot be empty.")
        if len(cleaned) > 100:
            raise ValidationError("Task title cannot exceed 100 characters.")
        return cleaned

    @staticmethod
    def _validate_due_date(due_date: str | None) -> str | None:
        if due_date in (None, ""):
            return None
        if not isinstance(due_date, str):
            raise ValidationError("Due date must be in YYYY-MM-DD format.")
        try:
            date.fromisoformat(due_date)
        except ValueError as exc:
            raise ValidationError("Due date must be a valid YYYY-MM-DD date.") from exc
        return due_date

    def _validate_priority(self, priority: str) -> str:
        if priority not in self.ALLOWED_PRIORITIES:
            allowed = ", ".join(sorted(self.ALLOWED_PRIORITIES))
            raise ValidationError(f"Priority must be one of: {allowed}.")
        return priority

    def _load(self) -> None:
        """Load tasks from JSON; a missing file means an empty task list."""
        if not self.storage_path.exists():
            self.tasks = []
            return
        try:
            with self.storage_path.open("r", encoding="utf-8") as file:
                raw_data: Any = json.load(file)
            if not isinstance(raw_data, list):
                raise ValueError("The JSON root must be a list.")
            self.tasks = [Task(**item) for item in raw_data]
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise StorageError(f"Could not load tasks: {exc}") from exc

    def save(self) -> None:
        """Save all tasks to the configured JSON file."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with self.storage_path.open("w", encoding="utf-8") as file:
                json.dump(
                    [asdict(task) for task in self.tasks],
                    file,
                    ensure_ascii=False,
                    indent=4,
                )
        except OSError as exc:
            raise StorageError(f"Could not save tasks: {exc}") from exc

    def add_task(
        self,
        title: str,
        description: str = "",
        priority: str = "normal",
        due_date: str | None = None,
    ) -> Task:
        """Validate and add a new task, then persist the change."""
        cleaned_title = self._validate_title(title)
        checked_priority = self._validate_priority(priority)
        checked_due_date = self._validate_due_date(due_date)
        next_id = max((task.task_id for task in self.tasks), default=0) + 1
        task = Task(
            task_id=next_id,
            title=cleaned_title,
            description=description.strip(),
            priority=checked_priority,
            due_date=checked_due_date,
        )
        self.tasks.append(task)
        self.save()
        return task

    def get_task(self, task_id: int) -> Task:
        """Return one task by its ID."""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        raise TaskNotFoundError(f"Task with ID {task_id} was not found.")

    def update_task(self, task_id: int, **changes: Any) -> Task:
        """Update supported task fields and persist the result."""
        task = self.get_task(task_id)
        allowed_fields = {"title", "description", "priority", "due_date"}
        unknown_fields = set(changes) - allowed_fields
        if unknown_fields:
            raise ValidationError(f"Unsupported fields: {', '.join(sorted(unknown_fields))}.")

        updated_values: dict[str, Any] = {}
        if "title" in changes:
            updated_values["title"] = self._validate_title(changes["title"])
        if "description" in changes:
            if not isinstance(changes["description"], str):
                raise ValidationError("Description must be a string.")
            updated_values["description"] = changes["description"].strip()
        if "priority" in changes:
            updated_values["priority"] = self._validate_priority(changes["priority"])
        if "due_date" in changes:
            updated_values["due_date"] = self._validate_due_date(changes["due_date"])

        for field, value in updated_values.items():
            setattr(task, field, value)
        self.save()
        return task

    def complete_task(self, task_id: int) -> Task:
        """Mark a task as completed."""
        task = self.get_task(task_id)
        task.completed = True
        self.save()
        return task

    def delete_task(self, task_id: int) -> None:
        """Delete a task by ID."""
        task = self.get_task(task_id)
        self.tasks.remove(task)
        self.save()

    def list_tasks(
        self,
        completed: bool | None = None,
        priority: str | None = None,
    ) -> list[Task]:
        """Return tasks filtered by completion state and/or priority."""
        if priority is not None:
            self._validate_priority(priority)
        result = self.tasks
        if completed is not None:
            result = [task for task in result if task.completed is completed]
        if priority is not None:
            result = [task for task in result if task.priority == priority]
        return list(result)

    def search_tasks(self, query: str) -> list[Task]:
        """Find tasks whose title or description contains the query."""
        if not isinstance(query, str):
            raise ValidationError("Search query must be a string.")
        normalized_query = query.strip().casefold()
        if not normalized_query:
            return []
        return [
            task
            for task in self.tasks
            if normalized_query in task.title.casefold()
            or normalized_query in task.description.casefold()
        ]

    def overdue_tasks(self, today: date | None = None) -> list[Task]:
        """Return incomplete tasks whose due date is earlier than today."""
        current_date = today or date.today()
        overdue: list[Task] = []
        for task in self.tasks:
            if task.completed or task.due_date is None:
                continue
            if date.fromisoformat(task.due_date) < current_date:
                overdue.append(task)
        return overdue
