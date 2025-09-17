from aiogram_dialog import Dialog

from .add_task import window as add_task_window
from .main_menu import window as main_menu_window


dialog = Dialog(
    add_task_window,
    main_menu_window,
)
