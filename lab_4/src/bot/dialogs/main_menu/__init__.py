from aiogram_dialog import Dialog

from .add_task import window as add_task_window
from .main_menu import window as main_menu_window
from .select_task import window as select_task_window
from .delete_or_edit_task import window as delete_or_edit_task_window
from .delete_task import window as delete_task_window


dialog = Dialog(
    add_task_window,
    main_menu_window,
    select_task_window,
    delete_or_edit_task_window,
    delete_task_window,
)
