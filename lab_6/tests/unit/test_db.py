import pytest
from typing import Any

from src.database import BaseDB, BaseDBMixin


class FakeDBMixin(BaseDBMixin):
    def create_user(self, tg_id: int, tasks_reversed: bool = False):
        self.users[tg_id] = {
            "tg_id": tg_id,
            "tasks_reversed": tasks_reversed,
        }
    
    def set_tasks_reversed(self, tg_id: int, value: bool):
        self.users[tg_id]["tasks_reversed"] = value

    def get_user(self, tg_id: int) -> dict[str, Any] | None:
        return self.users.get(tg_id, None)

    def add_task(self, tg_id: int, task: str):
        self.tasks.append({
            "user_id": tg_id,
            "task": task,
        })

    def get_task_by_idx(self, tg_id: int, idx: int) -> dict[str, Any] | None:
        tasks = self.get_tasks(tg_id)
        if idx > len(tasks) - 1:
            raise ValueError("Not found")
        
        return tasks[idx]

    def update_task_by_idx(self, tg_id: int, idx: int, new_task: str):
        curr_idx = -1
        for id_, task in enumerate(self.tasks):
            if task["user_id"] == tg_id:
                curr_idx += 1
                if curr_idx == idx:
                    self.tasks[id_]["task"] = new_task
                    return

        raise ValueError("Not found")

    def delete_task_by_idx(self, tg_id: int, idx: int):
        curr_idx = -1
        for id_, task in enumerate(self.tasks):
            if task["user_id"] == tg_id:
                curr_idx += 1
                if curr_idx == idx:
                    del self.tasks[id_]
                    return

        raise ValueError("Not found")

    def get_tasks_count(self, tg_id: int) -> int:
        return len(self.get_tasks(tg_id))

    def get_tasks(self, tg_id: int) -> list[dict[str, Any]]:
        if self.get_user(tg_id) is None:
            raise ValueError("User not found")

        return list(filter(lambda t: t["user_id"] == tg_id, self.tasks))


class FakeDB(BaseDB, FakeDBMixin):
    def __init__(self):
        self.users = {}
        self.tasks = []


@pytest.fixture
def db() -> BaseDB:
    fake = FakeDB()

    fake.create_user(1)
    fake.create_user(2)

    fake.set_tasks_reversed(2, True)
    
    fake.add_task(1, "Go walk")
    fake.add_task(2, "Cook")
    fake.add_task(1, "Sleep")
    fake.add_task(2, "Sport")
    fake.add_task(2, "Love")
    
    return fake


def test_create_user(db):
    tg_id = 3
    expected = {
        "tg_id": tg_id,
        "tasks_reversed": False,
    }

    db.create_user(tg_id)
    
    assert db.users[tg_id] == expected


def test_set_tasks_reversed(db):
    tg_id = 3
    tasks_reversed = True
    expected = {
        "tg_id": tg_id,
        "tasks_reversed": tasks_reversed
    }
    db.create_user(tg_id)

    db.set_tasks_reversed(tg_id, tasks_reversed)

    assert db.users[tg_id] == expected


def test_get_user(db):
    tg_id = 3
    expected = {
        "tg_id": tg_id,
        "tasks_reversed": False
    }
    db.create_user(tg_id)

    actual = db.get_user(tg_id)

    assert actual == expected


def test_add_task(db):
    tg_id = 3
    task = "Go walk"
    expected = {
        "user_id": tg_id,
        "task": task,
    }
    expected_tasks_count = len(db.tasks) + 1
    db.create_user(tg_id)

    db.add_task(tg_id, task)

    assert expected_tasks_count == len(db.tasks)
    assert db.tasks[-1] == expected


def test_get_task_by_idx(db):
    tg_id = 3
    task0 = "Go to date"
    task1 = "Go to hospital"
    db.create_user(tg_id)
    db.add_task(tg_id, task0)
    db.add_task(tg_id, task1)

    actual_task0 = db.get_task_by_idx(tg_id, idx=0)
    actual_task1 = db.get_task_by_idx(tg_id, idx=1)

    assert actual_task0["task"] == task0
    assert actual_task1["task"] == task1


def test_get_task_by_idx_raises_if_user_not_found(db):
    tg_id = 999
    db.create_user(tg_id)

    with pytest.raises(ValueError):
        db.get_task_by_idx(tg_id, 0)


def test_delete_task_by_idx(db):
    tg_id = 2
    idx = db.get_tasks_count(tg_id) - 1

    db.delete_task_by_idx(tg_id, idx)
    
    with pytest.raises(ValueError):
        db.delete_task_by_idx(tg_id, idx)


def test_delete_task_by_idx_raises_if_task_not_found(db):
    tg_id = 2
    idx = 999
    
    with pytest.raises(ValueError):
        db.delete_task_by_idx(tg_id, idx)


def test_update_task_by_idx(db):
    tg_id = 2
    idx = 1
    new_task = "Do lab6"
    expected = {
        "user_id": tg_id,
        "task": new_task,
    }

    db.update_task_by_idx(tg_id, idx, new_task)

    assert db.get_task_by_idx(tg_id, idx) == expected


def test_update_task_by_idx_raises_if_user_not_found(db):
    tg_id = 999
    idx = 0
    new_task = "Go gym"

    with pytest.raises(ValueError):
        db.update_task_by_idx(tg_id, idx, new_task)


def test_update_task_by_idx_raises_if_idx_not_found(db):
    tg_idx = 2
    idx = 999
    new_task = "Go sleep"

    with pytest.raises(ValueError):
        db.update_task_by_idx(tg_idx, idx, new_task)


def test_get_tasks(db):
    tg_id = 3
    task1 = "Go walk"
    task2 = "Cook"
    expected = [{"task": task, "user_id": tg_id} for task in (task1, task2)]
    db.create_user(tg_id)
    db.add_task(tg_id, task1)
    db.add_task(tg_id, task2)

    actual = db.get_tasks(tg_id)

    assert actual == expected


def test_get_tasks_raises_if_user_not_found(db):
    tg_id = 999

    with pytest.raises(ValueError):
        db.get_tasks(tg_id)
