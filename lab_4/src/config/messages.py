from src.utils import dict_to_sn


messages = dict_to_sn({
    "messages": {
        "enter_task": "Enter task:",
        "your_tasks": "Your tasks:",
    },
    "formats": {
        "tasks": "{item[0]}. {item[1]}",
    },
    "buttons": {
        "add": "Add",
        "delete": "Delete",
        "edit": "Edit",
        "edit_or_delete": "Edit / Delete",
        "back": "Back",
    },
    "errors": {
        "something_wrong": "Something went wrong :(",
    },
})
