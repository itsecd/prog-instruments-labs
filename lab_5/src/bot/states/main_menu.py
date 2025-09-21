from aiogram.fsm.state import State, StatesGroup


class MainMenuStatesGroup(StatesGroup):
    """States for main menu dialog"""
    choosing_action = State()
    adding_task = State()
    choosing_task = State()
    task_choosen = State()
    editing_task = State()
    deleting_task = State()
