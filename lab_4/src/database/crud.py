from typing import Any
from .db import get_connection


def create_user(tg_id: int, tasks_reversed: bool = False):
    conn = get_connection()
    conn.execute("INSERT INTO users (tg_id, tasks_reversed) VALUES (?, ?)", (tg_id, tasks_reversed))
    conn.commit()
    conn.close()


def get_user(tg_id: int) -> dict[str, Any]:
    conn = get_connection()
    cursor = conn.execute("SELECT * FROM users WHERE tg_id = ?", (tg_id, ))
    row = cursor.fetchone()

    conn.close()

    return dict(row)
