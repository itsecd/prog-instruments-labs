from aiogram import types, F
from aiogram.dispatcher.router import Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.methods.send_message import SendMessage
import keyboards
import consts
import lab_state

router = Router()


@router.message(Command(commands=["start"]))
async def start(msg: types.Message):
    if msg.from_user.id in consts.COUNCIL_MEMPERS_IDS:
        keyboard = keyboards.admin_keyboard()
    else:
        keyboard = keyboards.user_keyboard()
    await msg.answer(
        "Привет!\nЯ бот клуба Robotic и у меня можно узнать, открыта ли наша лаборатория.",
        reply_markup=keyboard,
    )


@router.message(F.text == "Открыть")
async def open_lab(msg: types.Message):
    if msg.from_user.id in consts.COUNCIL_MEMPERS_IDS:
        # answer = await lab_state.LabState.change_status(True)
        current_state = lab_state.LabState.get_state()
        if current_state:
            await msg.answer("Лаборатория уже открыта!")
        else:
            answer = lab_state.LabState.set_state(True)
            answer = (
                "Лаборатория переведена в состояние открыто."
                if answer
                else "Лаборатория переведена в состояние закрыто"
            )
            await msg.answer(answer)
    else:
        await msg.answer("У вас нет прав доступа.")


@router.message(F.text == "Закрыть")
async def close_lab(msg: types.Message):
    if msg.from_user.id in consts.COUNCIL_MEMPERS_IDS:
        # answer = await lab_state.LabState.change_status(True)
        current_state = lab_state.LabState.get_state()
        if not current_state:
            await msg.answer("Лаборатория уже закрыта!")
        else:
            answer = lab_state.LabState.set_state(False)
            answer = (
                "Лаборатория переведена в состояние открыто."
                if answer
                else "Лаборатория переведена в состояние закрыто"
            )
            await msg.answer(answer)
    else:
        await msg.answer("У вас нет прав доступа.")

@router.message(F.text == "Лаба открыта?")
async def get_lab_state(msg: types.Message):
    answer = lab_state.LabState.get_state()
    answer = (
                "Лаборатория открыта."
                if answer
                else "Лаборатория закрыта"
            )
    await msg.answer(answer)