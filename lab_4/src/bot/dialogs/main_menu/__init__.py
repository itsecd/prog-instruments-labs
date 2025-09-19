from aiogram_dialog import Dialog

from .choosing_action import window as choosing_action_window
from .adding_task import window as adding_task_window
from .choosing_task import window as choosing_task_window
from .task_choosen import window as task_choosen_window
from .deleting_task import window as deleting_task_window
from .editing_task import window as editing_task_window


dialog = Dialog(
    choosing_action_window,
    adding_task_window,
    choosing_task_window,
    task_choosen_window,
    deleting_task_window,
    editing_task_window,
)
