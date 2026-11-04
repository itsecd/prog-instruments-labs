"""Тесты сервиса задач с подменой репозитория и сессии на моки"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.repositories.task import TaskRepository
from app.schemas.task import TaskSchema, TaskCreateSchema, TaskUpdateSchema
from app.services.task import TaskService, TaskNotFound


@pytest.fixture
def db() -> Mock:
    """Mock сессии БД"""
    return Mock()


@pytest.fixture
def service(db: Mock) -> TaskService:
    """Сервис задач с подменой репозитория на Mock"""
    task_service = TaskService(db)
    task_service.task_repository = Mock(spec=TaskRepository)
    return task_service


def make_task(title: str = "Old title", completed: bool = False) -> SimpleNamespace:
    """Создаёт объект имитирующий задачу из БД"""
    return SimpleNamespace(id="task-1", title=title, completed=completed)


def test_create_task_saves_and_commits(service: TaskService, db: Mock) -> None:
    """Тест на создание задачи"""
    service.task_repository.create.return_value = make_task(title="1st task")
    result = service.create_task(TaskCreateSchema(title="1st task"))
    service.task_repository.create.assert_called_once_with(title="1st task")
    db.commit.assert_called_once()
    assert result == TaskSchema(id="task-1", title="1st task", completed=False)


def test_update_missing_task_raises_and_not_commit(
        service: TaskService,
        db: Mock
) -> None:
    """Тест на обновление несуществующей задачи"""
    service.task_repository.get_by_id.return_value = None
    with pytest.raises(TaskNotFound):
        service.update_task("missing", TaskUpdateSchema(title="New title"))
    db.commit.assert_not_called()


@pytest.mark.parametrize(
    ("update", "expected_title", "expected_completed"),
    [
        ({"title": "New title"}, "New title", False),
        ({"completed": True}, "Old title", True),
        ({"title": "New title", "completed": True}, "New title", True),
        ({}, "Old title", False),
    ],
)
def test_update_task_changes_only_given_fields(
    service: TaskService,
    db: Mock,
    update: dict[str, object],
    expected_title: str,
    expected_completed: bool,
) -> None:
    """Параметризованный тест"""
    task = make_task()
    service.task_repository.get_by_id.return_value = task
    result = service.update_task("task-1", TaskUpdateSchema(**update))

    assert task.title == expected_title
    assert task.completed == expected_completed
    assert result.title == expected_title
    assert result.completed == expected_completed
    db.commit.assert_called_once()


def test_delete_task_removes_found_task(service: TaskService, db: Mock) -> None:
    """Тест на удаление задачи по id единожды"""
    task = make_task()
    service.task_repository.get_by_id.return_value = task

    service.delete_task("task-1")

    service.task_repository.delete.assert_called_once_with(task)
    db.commit.assert_called_once()