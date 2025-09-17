from aiogram_dialog import Window, Dialog, DialogManager, StartMode, setup_dialogs
from aiogram_dialog.widgets.kbd import Button
from aiogram_dialog.widgets.text import Const

from src.bot.states.main_menu import MainMenuStatesGroup


dialog = Dialog(Window(
    Const("Your tasks:"), # R
    Button(Const("Add"), id="add_task"), # C
    Button(Const("Delete/Edit"), id="delete_or_edit_task"), # UD
    Button(Const("Reverse"), id="reverse_tasks"),
    state=MainMenuStatesGroup.main,
))
