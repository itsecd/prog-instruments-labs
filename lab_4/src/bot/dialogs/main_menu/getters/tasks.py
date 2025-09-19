from aiogram_dialog import DialogManager

from src.database.crud import get_tasks, get_user


async def tasks_getter(dialog_manager: DialogManager, **kwargs):
    user_id = dialog_manager.event.from_user.id
    user_tasks = map(lambda t: t["task"], get_tasks(user_id))

    user = get_user(user_id)
    if user["tasks_reversed"]:
        user_tasks = reversed(list(user_tasks))

    return {
        "tasks": list(enumerate(user_tasks, start=1)),
    }


async def task_getter(dialog_manager: DialogManager, **kwargs):
    user_id = dialog_manager.event.from_user.id
    user_tasks = get_tasks(user_id)

    task_idx = dialog_manager.dialog_data.get("task_idx")
    if task_idx is None or task_idx >= len(user_tasks):
        return {"task": "No task selected"}

    return {"task": user_tasks[task_idx]["task"]}
