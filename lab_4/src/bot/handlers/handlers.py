from aiogram import Router, types, F
from aiogram.filters import Command
from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.context import FSMContext

from bot import bot
from src.bot.data import data
import src.bot.keyboards.keyboards 
from src.bot.templ import templates

rt = Router()

#startcmd
@rt.message(Command('start'))
async def startcmd(message: types.Message):
    print("start rt")
    await message.answer(
        f'Приветствую {message.from_user.first_name}! \nЖелаешь воспользоваться таск-листом?',
        reply_markup=src.bot.keyboards.keyboards.startMarkup)

#id state
class idgetter(StatesGroup):
    userid = State()

#start callbacks
@rt.callback_query(F.data == 'startYes',)
async def startyes(callback: types.CallbackQuery, state: FSMContext):
    await callback.answer("Хорошо")
    
    #getting an id for idgetter
    await state.set_state(idgetter.userid)
    await state.set_data({'userid': callback.from_user.id}) 
    
    await callback.message.edit_text("Что будешь делать?", reply_markup=lab_4.src.bot.keyboards.keyboards.actionMarkup)

@rt.callback_query(F.data == 'startNo')
async def startno(callback: types.CallbackQuery):
    await callback.answer('Подумай...')

#task states
class adder(StatesGroup):
    userid = State()
    task = State()

class getter(StatesGroup):
    renderedtasks = State()

class deller(StatesGroup):
    userid = State()
    tasknums = State()

class editor(StatesGroup):
    userid = State()
    tasknums = State()
    editedtasks = State()

# some methods
async def getUserId(callback: types.CallbackQuery, state: FSMContext):
    currentstate = await state.get_state()
    if currentstate != idgetter.userid:
        await state.clear()
        await state.set_state(idgetter.userid)
    
    datafu = await state.get_data()
    if datafu == {}:
        await state.update_data(userid = callback.from_user.id)
        datafu = await state.get_data()
        
    userid = datafu.get('userid')
    await state.clear()
    print(f"userid in gui: {datafu}, {userid}")
    
    return userid

async def taskNumsChecker(message: types.Message, state: FSMContext, opname):
    if(int(message.text) <= (data.getCurNumOfTask(message.from_user.id)-1)):
        curtasknums = await state.get_data()
        curtasknums = curtasknums.get('tasknums')
        if curtasknums == None:
            await state.update_data(tasknums = [int(message.text)])
            await message.answer(f"Задание под номером {message.text} добавлено в список для {opname}")
            print(await state.get_data())
            return True
        elif isinstance(curtasknums, list) and int(message.text) not in curtasknums:
            curtasknums.append(int(message.text))
            await state.update_data(tasknums = curtasknums)
            await message.answer(f"Задание под номером {message.text} добавлено в список для {opname}")
            print(await state.get_data())
            return True
        elif int(message.text) in curtasknums:
            await message.answer(f"Задание под номером {message.text} уже добавлено в список для {opname}")
            return False
        else:
            print("error")
            print(await state.get_data())
            return False
    else:
        await message.answer("Задание не найдено")
        return False

#task callback handlers
#add tasks
@rt.callback_query(F.data == 'addTask')
async def addTask(callback: types.CallbackQuery, state: FSMContext):
    
    #getting userid for case
    currentstate = await state.get_state()
    if currentstate != idgetter.userid:
        await state.set_state(idgetter.userid)
        
    datafu = await state.get_data()
    if datafu == {}:
        print("id updated in state")
        await state.update_data(userid = callback.from_user.id)
        datafu = await state.get_data()
    
    userid = datafu.get('userid')
    print(f"userid: {datafu}")
    
    #puting userid in adder object
    await state.set_state(adder.userid)
    await state.update_data(userid = userid)
    
    await state.set_state(adder.task)
    await callback.message.edit_text(f'Отлично!\nЭто будет задание под номером {data.getCurNumOfTask(userid)}\nНапиши своё задание:')

@rt.message(adder.task)
async def taskAdder(message: types.Message, state: FSMContext):
    #getting a 'task' for adder object
    await state.update_data(task=message.text)
    
    datafu = await state.get_data()                
    await state.clear()
    
    result = data.taskAdder(datafu)
    if result:
        await message.answer(f"Отлично!\nВы пополили свой таск-лист заданием: {datafu['task']}\nЧто делаем дальше?",
                             reply_markup=lab_4.src.bot.keyboards.keyboards.adderMarkup)

@rt.callback_query(F.data == 'addanother')
async def taskadderanother(callback: types.CallbackQuery, state: FSMContext):
    await addTask(callback, state)
            
#get tasks
@rt.callback_query(F.data == 'getTask')
async def getTask(callback: types.CallbackQuery, state: FSMContext):
    userid = await getUserId(callback, state)
    tasks = data.getUserTasks(userid)
    
    if tasks != []:
        await callback.answer('')
        await callback.message.edit_text(
            f'Взгляни на свои задания!\nТут {data.getCurNumOfTask(userid) - 1} заданий:\n\n{templates.RenderTaskList(tasks)}\n\nЧто будешь делать?',
            reply_markup=lab_4.src.bot.keyboards.keyboards.getterMarkup)
    else:
        await callback.answer('Нет заданий')
        
@rt.callback_query(F.data == 'hideTask')
async def hidetask(callback: types.CallbackQuery, state: FSMContext):
    await startyes(callback, state)

#del tasks
@rt.callback_query(F.data == 'delTask')
async def delTask(callback: types.CallbackQuery, state: FSMContext):
    userid = await getUserId(callback, state) 
    tasks = data.getUserTasks(userid)
    if(tasks == []):
        await callback.answer('Нет заданий')
        await deleditTaskBack(callback,state)
    await state.set_state(getter.renderedtasks)
    renderedtasks = templates.RenderTaskList(tasks)
    await state.update_data(renderedtasks = renderedtasks)
    
    await callback.message.edit_text(f'!Режим удаления!\nТвой таск-лист сейчас:\n\n{renderedtasks}\nЧто будем делать?', reply_markup=lab_4.src.bot.keyboards.keyboards.dellerMarkup)

@rt.callback_query(F.data == 'TaskDeller')
async def TasksDeller(callback: types.CallbackQuery, state: FSMContext):
    await state.set_state(getter.renderedtasks)
    renderedtasks = await state.get_data()
    
    renderedtasks = renderedtasks.get('renderedtasks') 
    userid = await getUserId(callback, state)
    
    await state.set_state(deller.userid)
    await state.update_data(userid = userid)
    print(await state.get_data())
    await state.set_state(deller.tasknums)
    
    await bot.delete_message(callback.message.chat.id, callback.message.message_id)
    await callback.message.answer(f'!Режим удаления!\nТвой таск-лист сейчас:\n\n{renderedtasks}\nВыбери или напиши номера существующих заданий для удаления:',
                                  reply_markup=lab_4.src.bot.keyboards.keyboards.taskskbbuilder(userid,"подтвердить"))
    
@rt.message(deller.tasknums, F.text.isdigit())
async def addtaskindeller(message: types.Message, state: FSMContext):
    await taskNumsChecker(message, state, "удаления")
                
@rt.message(deller.tasknums, F.text == "подтвердить")
async def senddeller(message: types.Message, state: FSMContext):
    datafu = await state.get_data()
    print(datafu)
    if 'tasknums' in datafu and datafu['tasknums'] is not None:
        print("accepted")                
        await state.clear()
        
        result = data.deleteTasks(datafu)
        if result: 
            await bot.send_message(message.chat.id,"kbdel" , reply_markup=lab_4.src.bot.keyboards.keyboards.ReplyKeyboardRemove())
            await bot.delete_message(message.chat.id, message.message_id + 1)
            await bot.delete_message(message.chat.id, message.message_id)
            await bot.send_message(datafu['userid'], "Отлично!\nНадеюсь вы спарвились с заданиями)",reply_markup=lab_4.src.bot.keyboards.keyboards.dellerMarkupback)
    else: 
        await message.answer("Вы не выбрали задания")

@rt.callback_query(F.data == 'clearTask')
async def clearTasks(callback: types.CallbackQuery, state: FSMContext):
    userid = await getUserId(callback, state)
    result = data.clearTasks(userid)
    if result:
        await callback.answer("Таск-лист очищен")
        await state.clear()
        await startyes(callback, state)
    else:
        await callback.answer("Ошибка :(")
        await state.clear()
        await startyes(callback, state)
   
#edit tasks
@rt.callback_query(F.data == 'editTask')
async def editTask(callback: types.CallbackQuery, state: FSMContext):
    userid = await getUserId(callback, state)
    tasks = data.getUserTasks(userid)
    if(tasks == []):
        await callback.answer('Нет заданий')
        await deleditTaskBack(callback,state)
    await state.set_state(getter.renderedtasks)
    renderedtasks = templates.RenderTaskList(tasks)
    await state.update_data(renderedtasks = renderedtasks)
    
    await callback.message.edit_text(f'!Режим редактирования!\nТвой таск-лист сейчас:\n\n{renderedtasks}\nЧто будем делать?', reply_markup=lab_4.src.bot.keyboards.keyboards.editorMarkup)

@rt.callback_query(F.data == 'TaskEditor')
async def TaskEditor(callback: types.CallbackQuery, state: FSMContext):
    await state.set_state(getter.renderedtasks)
    renderedtasks = await state.get_data()
    
    renderedtasks = renderedtasks.get('renderedtasks') 
    userid = await getUserId(callback, state)
    
    await state.set_state(editor.userid)
    await state.update_data(userid = userid)
    print(await state.get_data())
    await state.set_state(editor.tasknums)
    
    await bot.delete_message(callback.message.chat.id, callback.message.message_id)
    await callback.message.answer(f'!Режим редактирования!\nТвой таск-лист сейчас:\n\n{renderedtasks}\nВыбери или напиши номер существующего заданий и напиши его заново:',
                                  reply_markup=lab_4.src.bot.keyboards.keyboards.taskskbbuilder(userid,"сохранить изменения"))
    
@rt.message(editor.tasknums, F.text.isdigit())
async def addtaskineditor(message: types.Message, state: FSMContext):
    res = await taskNumsChecker(message, state, "редактирования")
    print(res)
    if res:
        await state.set_state(editor.editedtasks)
        await message.answer(f"Выбрано задание под номером {message.text}\nНапишите на какое задание хотите его изменить:", reply_markup=lab_4.src.bot.keyboards.keyboards.ReplyKeyboardRemove())
    else:
        return 

@rt.message(editor.editedtasks)
async def editedtask(message: types.Message, state: FSMContext):
    currenttasks = await state.get_data()
    currenttasks = currenttasks.get('editedtasks')
    if currenttasks == None:
        await state.update_data(editedtasks = [message.text])
        await message.answer(f"Задание: {message.text} записано для редактирования",reply_markup=lab_4.src.bot.keyboards.keyboards.taskskbbuilder(message.from_user.id,"сохранить изменения"))
    elif isinstance(currenttasks, list):
        currenttasks.append(message.text)
        await state.update_data(editedtasks = currenttasks)
        await message.answer(f"Задание: {message.text} записано для редактирования",reply_markup=lab_4.src.bot.keyboards.keyboards.taskskbbuilder(message.from_user.id,"сохранить изменения"))
    else:
        print("error")
    print(await state.get_data())
    await state.set_state(editor.tasknums)
        
@rt.message(editor.tasknums, F.text == 'сохранить изменения')
async def sendeditor(message: types.Message, state: FSMContext):
    datafu = await state.get_data()
    print(datafu)
    if 'tasknums' in datafu and datafu['tasknums'] is not None and 'editedtasks' in datafu and datafu['editedtasks'] is not None:
        print("accepted")
        await state.clear()
        result = data.editTasks(datafu)
        if result: 
            await bot.send_message(message.chat.id,"kbdel" , reply_markup=lab_4.src.bot.keyboards.keyboards.ReplyKeyboardRemove())
            await bot.delete_message(message.chat.id, message.message_id + 1)
            await bot.delete_message(message.chat.id, message.message_id)
            await bot.send_message(datafu['userid'], "Отлично!\nЗадания отредактированы)",reply_markup=lab_4.src.bot.keyboards.keyboards.dellerMarkupback)
    else: 
        await message.answer("Вы не выбрали задания")
    
@rt.callback_query(F.data == 'reverseTask')
async def reverseTask(callback: types.CallbackQuery, state: FSMContext):
    userid = await getUserId(callback, state)
    result = data.reverseTasks(userid)
    if result:
        await callback.answer("Таск-лист перевёрнут успешно)")
        await state.clear()
        await startyes(callback, state)
    else:
        await callback.answer("Ошибка :(")
        await state.clear()
        await startyes(callback, state)
    
# edit/delete task back    
@rt.callback_query(F.data == 'deleditTaskBack')
async def deleditTaskBack(callback: types.CallbackQuery,state: FSMContext):
    await state.clear()
    await startyes(callback, state)    