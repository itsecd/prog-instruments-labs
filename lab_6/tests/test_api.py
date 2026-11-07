"""Тесты API с тестовой базой SQLite"""

import pytest
from fastapi.testclient import TestClient


def test_create_task_add_to_list(client: TestClient) -> None:
    """Созданная задача возвращается с кодом 201 и добавляется в список"""
    response = client.post("/tasks", json={"title": "1st task"})

    assert response.status_code == 201
    task = response.json()
    assert task["title"] == "1st task"
    assert task["completed"] is False
    assert client.get("/tasks").json() == [task]


def test_complete_task_via_patch(client: TestClient) -> None:
    """Отметка задачи как выполненная без изменения названия"""
    task_id = client.post("/tasks", json={"title": "1st task"}).json()["id"]

    response = client.patch(f"/tasks/{task_id}", json={"completed": True})

    assert response.status_code == 200
    assert response.json() == {
        "id": task_id,
        "title": "1st task",
        "completed": True,
    }


def test_delete_task_remove_from_list(client: TestClient) -> None:
    """Удаляемая задача удаляется из списка, возвращается код 204"""
    task_id = client.post("/tasks", json={"title": "Temp"}).json()["id"]

    response = client.delete(f"/tasks/{task_id}")

    assert response.status_code == 204
    assert client.get("/tasks").json() == []


def test_create_and_rename_category(client: TestClient) -> None:
    """Создание категории, возврат кода 201 и переименование"""
    create_response = client.post("/categories", json={"name": "Category 1"})
    assert create_response.status_code == 201
    category_id = create_response.json()["id"]

    response = client.patch(f"/categories/{category_id}", json={"name": "Category 2"})

    assert response.status_code == 200
    assert response.json() == {"id": category_id, "name": "Category 2"}
    assert client.get("/categories").json() == [response.json()]


def test_delete_category_remove_from_list(client: TestClient) -> None:
    """Удаляемая категория удаляется из списка, возвращается код 204"""
    category_id = client.post("/categories", json={"name": "Temp"}).json()["id"]

    response = client.delete(f"/categories/{category_id}")

    assert response.status_code == 204
    assert client.get("/categories").json() == []


@pytest.mark.parametrize(
    "payload",
    [{}, {"title": ""}, {"title": None}, {"name": "Wrong field"}],
    ids=["no title", "empty title", "None title", "wrong field"]
)
def test_create_task_with_invalid_payload(
    client: TestClient,
    payload: dict[str, object],
) -> None:
    """Тест обработки некорректного тела запроса"""
    response = client.post("/tasks", json=payload)

    assert response.status_code == 422


@pytest.mark.parametrize(
    ("method", "url", "payload"),
    [
        ("PATCH", "/tasks/missing", {"title": "New title"}),
        ("DELETE", "/tasks/missing", None),
        ("PATCH", "/categories/missing", {"name": "New name"}),
        ("DELETE", "/categories/missing", None),
    ],
    ids=["patch task", "delete task", "patch category", "delete category"]
)
def test_missing_resource(
    client: TestClient,
    method: str,
    url: str,
    payload: dict[str, str] | None
) -> None:
    """Обращение к несуществующему объекту, возвращается код 404"""
    response = client.request(method, url, json=payload)

    assert response.status_code == 404
