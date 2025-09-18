from typing import Any
from .db import get_connection


def create_user(tg_id: int, tasks_reversed: bool = False):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO users (tg_id, tasks_reversed) VALUES (?, ?)",
            (tg_id, int(tasks_reversed))
        )
        conn.commit()


def set_tasks_reversed(tg_id: int, value: bool):
    with get_connection() as conn:
        conn.execute(
            "UPDATE users SET tasks_reversed = ? WHERE tg_id = ?",
            (int(value), tg_id)
        )
        conn.commit()


def get_user(tg_id: int) -> dict[str, Any]:
    with get_connection() as conn:
        cursor = conn.execute(
            "SELECT tg_id, tasks_reversed FROM users WHERE tg_id = ?",
            (tg_id, )
        )
        row = cursor.fetchone()

    return dict(row)


def add_task(tg_id: int, task: str):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO tasks (user_id, task) VALUES (?, ?)",
            (tg_id, task)
        )
        conn.commit()


def get_task_by_idx(tg_id: int, idx: int) -> dict[str, Any]:
    with get_connection() as conn:
        cursor = conn.execute(
            "SELECT * from tasks WHERE user_id = ? ORDER BY id LIMIT 1 OFFSET ?",
            (tg_id, idx, )
        )
        row = cursor.fetchone()

    return dict(row)


def update_task_by_idx(tg_id: int, idx: int, new_task: str):
    task = get_task_by_idx(tg_id, idx)
    with get_connection() as conn:
        conn.execute(
            "UPDATE tasks SET task = ? WHERE id = ?",
            (new_task, task["id"])
        )
        conn.commit()


def delete_task_by_idx(tg_id: int, idx: int):
    task = get_task_by_idx(tg_id, idx)
    with get_connection() as conn:
        conn.execute(
            "DELETE FROM tasks WHERE id = ?",
            (task["id"], )
        )
        conn.commit()


def get_tasks(tg_id: int) -> list[str]:
    with get_connection() as conn:
        cursor = conn.execute(
            "SELECT * FROM tasks WHERE user_id = ? ORDER BY id",
            (tg_id, )
        )
        rows = cursor.fetchall()

    return [dict(task) for task in rows]
