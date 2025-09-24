from aiogram.types import CallbackQuery
from aiogram.fsm.state import State

from aiogram_dialog import DialogManager
from aiogram_dialog.widgets.kbd import Button


def goto_state(state: State):
    """
    Generates a handler to handle button clicks to avoid repetitions

    Args:
        state (State): necessary state
    """
    async def handler(
        callback: CallbackQuery,
        button: Button,
        dialog_manager: DialogManager
    ):
        await dialog_manager.switch_to(state)

    return handler
