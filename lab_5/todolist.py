import json
import os
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

filename = "todo.json"
if os.path.exists(filename) and os.path.getsize(filename) > 0:
    with open(filename, "r") as file:
        tasks = json.load(file)
else:
    tasks = []

def add_task():
    n= int(input("Enter the no of tasks:"))
    for i in range(n):
        print(f"The details of task {i+1} :")
        Tasknumber = input("Enter the tasknumber:")
        if  not Tasknumber.isdigit():
            while(True):
                Tasknumber = (input("Enter a valid combination of the nos(0-9):"))
                if Tasknumber.isdigit():
                    Tasknumber =int(Tasknumber)
                    break
        Description = input("Enter the name :")
        if not Description.isalpha():
            while(True):
                Description = input("Enter a valid combination of the letters(A-Z):")
                if Description.isalpha():
                    break
        Priority = input("Enter the priority(low\\medium\\high):").lower()
        if Priority not in ["low","medium","high"]:
            while(True):
                Priority= input("Enter a valid one:")
                if Priority in ["low","medium","high"]:
                    break
        Duedate = input("Enter the due date eg.(2025-08-29)")
        while(True):
            try:
                dt = datetime.strptime(Duedate,"%d-%m-%Y").date()
                Duedate = dt.strftime("%d-%m-%Y")
                break
            except ValueError:
                Duedate = input("Enter a valid due date:")
        tasks.append({"Tasknumber":Tasknumber,"Description":Description,"Priority":Priority,"Duedate":Duedate,"status":False})
        logger.info("Task added: number=%s, priority=%s", Tasknumber, Priority)
def display_tasks(uw):
    if uw not in ['all','hp']:
        while(True):
            uw = input("Enter a valid option(all\\hp):")
            if uw in ['all','hp']:
                break
    if uw == 'all':
        logger.info("Displaying all tasks: count=%s", len(tasks))
        for task in tasks:
            print(f"Taskno:{task['Tasknumber']},Description:{task['Description']},Priority:{task['Priority']},Duedate:{task['Duedate']},Status:{task['status']}")
    else:
        highprioritytasks = [task for task in tasks if task['Priority'] == 'high']
        logger.info("Displaying high-priority tasks: count=%s", len(highprioritytasks))
        for task in highprioritytasks:
            print(f"Taskno:{task['Tasknumber']},Description:{task['Description']},Priority:{task['Priority']},Duedate:{task['Duedate']},Status:{task['status']}")
def update_task(uw):
    if uw == 's':
        c = int(input("Enter the tasknumber you want to change the status:"))
        for task in tasks:
            if c == int(task['Tasknumber']):
                print('hi')
                task['status'] = True
                logger.info("Task status updated: number=%s, status=%s", c, task['status'])
                break
    elif uw == 'x':
        c = int(input("Enter the tasknumber to delete that task:"))
        for task in tasks:
            if c == int(task['Tasknumber']):
                tasks.remove(task)
                logger.info("Task deleted: number=%s", c)
                break
    else:
        c = input("Enter the entitity you want to modify in a task(e.g Due date,Description) :").lower()
        if c not in ['dd','des']:
            while(True):
                c = input("Enter a valid option(dd\\des)")
                if c in ['dd','des']:
                    break
        c1 = int(input("Enter its tasknumber:"))
        if c == 'dd':
            Duedaten = input("Enter the due date eg.(25-08-2025)")
            while(True):
                try:
                    dt = datetime.strptime(Duedaten,"%d-%m-%Y").date()
                    Duedaten = dt.strftime("%d-%m-%Y")
                    break
                except ValueError:
                    Duedaten = input("Enter a valid due date:")
            for task in tasks:
                if c1 == int(task['Tasknumber']):
                    task['Duedate'] = Duedaten
                    logger.info("Task due date updated: number=%s, new_date=%s", c1, Duedaten)
                    break
        else:
            for task in tasks:
                if c1 == int(task['Tasknumber']):
                    Newdescription = input("Enter the new description")
                    task['Description'] = Newdescription
                    logger.info("Task description updated: number=%s", c1)
                    break
def saver():
    with open(filename, "w") as file:
        json.dump(tasks, file, indent=2)
    logger.info("Tasks saved: file=%s, count=%s", filename, len(tasks))

logger.info("Application started")

while(True):    
    uc = input("what do you want to do\nAdd task(a)\nDisplay tasks(d)\nUpdate task(u)\nExit(e)\n(Save the file)(l)").lower()
    if uc not in ['a','d','u','e','l']:
        while(True):
            logger.warning("Invalid menu option: %r", uc)
            uc = input("Choose a valid option")
            if uc in ['a','d','u','e','l']:
                break
    if uc == 'a':
        add_task()
    if uc == 'd':
        uw  = input("Do you want to display all the tasks(all) (or)\n(Only the tasks with  high priority)(high):")
        display_tasks(uw)
    if uc == 'u':
        choice = input("Do you want to\nchange the status by number(s)\nDelete a task by number(x)\nEdit a task((et)")
        update_task(choice)
    if uc == 'l':
        saver()
    if uc == 'e':
        print("Finish tasks on time")
        break
