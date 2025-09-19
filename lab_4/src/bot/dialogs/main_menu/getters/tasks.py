from aiogram_dialog import DialogManager

from src.database.crud import get_tasks, get_user
from src.utils import limit_string


async def tasks_getter(dialog_manager: DialogManager, **kwargs) -> dict[str, list[int, str]]:
    user_id = dialog_manager.event.from_user.id
    user = get_user(user_id)
    user_tasks = get_tasks(user_id)

    tasks_texts = map(
        lambda t: t["task"],
        user_tasks,
    )

    if user["tasks_reversed"]:
        tasks_texts = reversed(list(tasks_texts))

    tasks = list(enumerate(tasks_texts, start=1))
    tasks_for_buttons = [(i, limit_string(t)) for i, t in tasks]

    return {
        "tasks": tasks,
        "tasks_for_buttons": tasks_for_buttons,
    }


async def task_getter(dialog_manager: DialogManager, **kwargs) -> dict[str, str] | None:
    user_id = dialog_manager.event.from_user.id
    user_tasks = get_tasks(user_id)

    task_idx = dialog_manager.dialog_data.get("task_idx")
    if task_idx is None or task_idx >= len(user_tasks):
        return None

    return {
        "task": user_tasks[task_idx]["task"]
    }
