#!/usr/bin/env python3
import sys
import json
import os
from datetime import datetime

TASKS_FILE = "tasks.json"

def load_tasks():
    if not os.path.exists(TASKS_FILE):
        return []
    with open(TASKS_FILE, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []

def save_tasks(tasks):
    with open(TASKS_FILE, "w") as f:
        json.dump(tasks, f, indent=4)

def generate_id(tasks):
    return max([task["id"] for task in tasks], default=0) + 1

def add_task(description):
    tasks = load_tasks()
    task = {
        "id": generate_id(tasks),
        "description": description,
        "status": "todo",
        "createdAt": datetime.now().isoformat(),
        "updatedAt": datetime.now().isoformat(),
    }
    tasks.append(task)
    save_tasks(tasks)
    print(f"Task added successfully (ID: {task['id']})")

def update_task(task_id, description):
    tasks = load_tasks()
    for task in tasks:
        if task["id"] == task_id:
            task["description"] = description
            task["updatedAt"] = datetime.now().isoformat()
            save_tasks(tasks)
            print("Task updated successfully")
            return
    print("Task not found")

def delete_task(task_id):
    tasks = load_tasks()
    new_tasks = [task for task in tasks if task["id"] != task_id]
    if len(tasks) == len(new_tasks):
        print("Task not found")
    else:
        save_tasks(new_tasks)
        print("Task deleted successfully")

def mark_status(task_id, status):
    tasks = load_tasks()
    for task in tasks:
        if task["id"] == task_id:
            task["status"] = status
            task["updatedAt"] = datetime.now().isoformat()
            save_tasks(tasks)
            print(f"Task marked as {status}")
            return
    print("Task not found")

def list_tasks(filter_status=None):
    tasks = load_tasks()
    if filter_status:
        tasks = [task for task in tasks if task["status"] == filter_status]

    if not tasks:
        print("No tasks found")
        return

    for task in tasks:
        print(f"[{task['id']}] {task['description']} - {task['status']}")

def main():
    if len(sys.argv) < 2:
        print("Usage: task-cli <command> [arguments]")
        return

    command = sys.argv[1]

    if command == "add":
        if len(sys.argv) < 3:
            print("Usage: task-cli add <description>")
            return
        add_task(" ".join(sys.argv[2:]))

    elif command == "update":
        if len(sys.argv) < 4:
            print("Usage: task-cli update <id> <description>")
            return
        update_task(int(sys.argv[2]), " ".join(sys.argv[3:]))

    elif command == "delete":
        if len(sys.argv) < 3:
            print("Usage: task-cli delete <id>")
            return
        delete_task(int(sys.argv[2]))

    elif command == "mark-in-progress":
        if len(sys.argv) < 3:
            print("Usage: task-cli mark-in-progress <id>")
            return
        mark_status(int(sys.argv[2]), "in-progress")

    elif command == "mark-done":
        if len(sys.argv) < 3:
            print("Usage: task-cli mark-done <id>")
            return
        mark_status(int(sys.argv[2]), "done")

    elif command == "list":
        if len(sys.argv) == 2:
            list_tasks()
        else:
            status = sys.argv[2]
            if status not in ["todo", "in-progress", "done"]:
                print("Invalid status. Use: todo, in-progress, or done")
            else:
                list_tasks(status)

    else:
        print("Unknown command")

if __name__ == "__main__":
    main()