from typing import Any
from .db import get_connection


def create_user(tg_id: int, tasks_reversed: bool = False):
    """
    Creates a user record in the database

    Args:
        tg_id (int): telegram user id
        tasks_reversed (bool, optional): order of tasks. Defaults to False.
    """
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO users (tg_id, tasks_reversed) VALUES (?, ?)",
            (tg_id, int(tasks_reversed))
        )
        conn.commit()


def set_tasks_reversed(tg_id: int, value: bool):
    """
    Sets the order of tasks

    Args:
        tg_id (int): telegram user id
        value (bool)
    """
    with get_connection() as conn:
        conn.execute(
            "UPDATE users SET tasks_reversed = ? WHERE tg_id = ?",
            (int(value), tg_id)
        )
        conn.commit()


def get_user(tg_id: int) -> dict[str, Any] | None:
    """
    Gets a user from the database

    Args:
        tg_id (int): telegram user id

    Returns:
        dict[str, Any] | None: user data
    """
    with get_connection() as conn:
        cursor = conn.execute(
            "SELECT tg_id, tasks_reversed FROM users WHERE tg_id = ?",
            (tg_id, )
        )
        row = cursor.fetchone()

    return dict(row) if row is not None else None


def add_task(tg_id: int, task: str):
    """
    Adds a task to the database 

    Args:
        tg_id (int): telegram user id
        task (str): task text
    """
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO tasks (user_id, task) VALUES (?, ?)",
            (tg_id, task)
        )
        conn.commit()


def get_task_by_idx(tg_id: int, idx: int) -> dict[str, Any] | None:
    """
    Gets a task by its idx from the database

    Args:
        tg_id (int): telegram user id
        idx (int): task idx by order

    Returns:
        dict[str, Any] | None: task data
    """
    with get_connection() as conn:
        cursor = conn.execute(
            "SELECT * from tasks WHERE user_id = ? ORDER BY id LIMIT 1 OFFSET ?",
            (tg_id, idx, )
        )
        row = cursor.fetchone()

    return dict(row) if row is not None else None


def update_task_by_idx(tg_id: int, idx: int, new_task: str):
    """
    Updates a task by idx in the database

    Args:
        tg_id (int): telegram user id
        idx (int): task idx by order
        new_task (str): new task text
    """
    task = get_task_by_idx(tg_id, idx)
    with get_connection() as conn:
        conn.execute(
            "UPDATE tasks SET task = ? WHERE id = ?",
            (new_task, task["id"])
        )
        conn.commit()


def delete_task_by_idx(tg_id: int, idx: int):
    """
    Deletes a task by its idx from the database

    Args:
        tg_id (int): telegram user id
        idx (int): task idx by order
    """
    task = get_task_by_idx(tg_id, idx)
    with get_connection() as conn:
        conn.execute(
            "DELETE FROM tasks WHERE id = ?",
            (task["id"], )
        )
        conn.commit()


def get_tasks_count(tg_id: int) -> int:
    """
    Gets tasks count from the database

    Args:
        tg_id (int): telegram user id

    Returns:
        int: tasks count
    """
    with get_connection() as conn:
        cursor = conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE user_id = ?",
            (tg_id, )
        )
        row = cursor.fetchone()

    return row[0] if row else 0


def get_tasks(tg_id: int) -> list[dict[str, Any]]:
    """
    Gets all tasks for user from the database 

    Args:
        tg_id (int): telegram user id

    Returns:
        list[dict[str, Any]]: list of tasks data
    """
    with get_connection() as conn:
        cursor = conn.execute(
            "SELECT * FROM tasks WHERE user_id = ? ORDER BY id",
            (tg_id, )
        )
        rows = cursor.fetchall()

    return [dict(task) for task in rows]
