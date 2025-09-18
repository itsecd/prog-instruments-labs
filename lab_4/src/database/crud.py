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


def get_user(tg_id: int) -> dict[str, Any] | None:
    with get_connection() as conn:
        cursor = conn.execute("SELECT * FROM users WHERE tg_id = ?", (tg_id, ))
        row = cursor.fetchone()

    return dict(row) if row else None
