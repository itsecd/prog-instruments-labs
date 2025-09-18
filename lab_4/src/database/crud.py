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


def get_user(tg_id: int) -> tuple[int, bool]:
    with get_connection() as conn:
        cursor = conn.execute(
            "SELECT tg_id, tasks_reversed FROM users WHERE tg_id = ?",
            (tg_id, )
        )
        row = cursor.fetchone()

    if not row:
        raise RuntimeError(f"No user with tg_id = {tg_id}")

    return dict(row)


def add_task(tg_id: int, task: str):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO tasks (user_id, task) VALUES (?, ?)",
            (tg_id, task)
        )
        conn.commit()


def get_tasks(tg_id: int) -> list[str]:
    with get_connection() as conn:
        cursor = conn.execute(
            "SELECT task FROM tasks WHERE user_id = ?",
            (tg_id, )
        )
        rows = cursor.fetchall()

    if not rows:
        raise RuntimeError(f"No tasks for tg_id = {tg_id}")

    return [d["task"] for d in list(rows)]
