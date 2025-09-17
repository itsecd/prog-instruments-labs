from aiogram.fsm.state import State, StatesGroup


class MainMenuStatesGroup(StatesGroup):
    main = State()
    adding_task = State()
    selecting_task = State()
    editing_task = State()
    deleting_task = State()
    reversing_task = State()
