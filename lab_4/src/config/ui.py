from src.utils import dict_to_sn


ui = dict_to_sn({
    "messages": {
        "enter_task": "Enter task:",
        "enter_new_task": "Enter new task:",
        "your_tasks": "Your tasks:",
        "choose_task": "Choose task:",
        "choose_action": "Choose action:",
        "ask_delete": "Delete?",
    },
    "formats": {
        "tasks": "{item[0]}. {item[1]}",
    },
    "buttons": {
        "add": "Add",
        "delete": "Delete",
        "edit": "Edit",
        "edit_or_delete": "Edit / Delete",
        "reverse": "Reverse",
        "back": "Back",
    },
    "errors": {
        "something_wrong": "Something went wrong :(",
    },
})
