from aiogram.types import CallbackQuery

from aiogram_dialog import Window, DialogManager
from aiogram_dialog.widgets.kbd import Button
from aiogram_dialog.widgets.text import Const, List, Format

from src.bot.states.main_menu import MainMenuStatesGroup


# for tests
tasks = {}


async def do_add(callback: CallbackQuery, button: Button, dialog_manager: DialogManager):
    await dialog_manager.switch_to(MainMenuStatesGroup.adding_task)


async def do_reverse(callback: CallbackQuery, button: Button, dialog_manager: DialogManager):
    tasks[callback.from_user.id].reverse()
    await callback.answer(text="Reversed!")
    await dialog_manager.switch_to(MainMenuStatesGroup.main)


async def do_select(callback: CallbackQuery, button: Button, dialog_manager: DialogManager):
    if len(tasks.get(callback.from_user.id, [])) == 0:
        await callback.answer("No tasks!")
        return

    await dialog_manager.switch_to(MainMenuStatesGroup.selecting_task)
    

async def get_tasks(dialog_manager: DialogManager, **kwargs):
    user_id = dialog_manager.event.from_user.id
    user_tasks = tasks.get(user_id, [])

    return {
        "tasks": list(enumerate(user_tasks, start=1))
    }


window = Window(
    Const("Your tasks:"),
    List(
        Format("{item[0]}\. {item[1]}"),
        items="tasks",
    ),
    Button(Const("Add"), id="add_task", on_click=do_add), # C
    Button(Const("Delete/Edit"), id="delete_or_edit_task", on_click=do_select), # UD
    Button(Const("Reverse"), id="reverse_tasks", on_click=do_reverse),
    state=MainMenuStatesGroup.main,
    getter=get_tasks,
)
