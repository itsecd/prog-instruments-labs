#!/usr/bin/env python3
import sys
import json
import logging
import os
from datetime import datetime

from logging_setup import setup_logging

TASKS_FILE = "tasks.json"

logger = logging.getLogger("task_cli")

def load_tasks():
    if not os.path.exists(TASKS_FILE):
        logger.info("Tasks file %s not found, starting with an empty list", TASKS_FILE)
        return []
    with open(TASKS_FILE, "r") as f:
        try:
            tasks = json.load(f)
        except json.JSONDecodeError as exc:
            logger.error(
                "Tasks file %s is corrupted (%s), using an empty list", TASKS_FILE, exc
            )
            return []
    logger.debug("Loaded %d tasks from %s", len(tasks), TASKS_FILE)
    return tasks


def save_tasks(tasks):
    try:
        with open(TASKS_FILE, "w") as f:
            json.dump(tasks, f, indent=4)
    except OSError:
        logger.exception("Failed to save tasks to %s", TASKS_FILE)
        raise
    logger.debug("Saved %d tasks to %s", len(tasks), TASKS_FILE)

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
    logger.info(
        "Task added: id=%d, description_length=%d", task["id"], len(description)
    )
    print(f"Task added successfully (ID: {task['id']})")


def update_task(task_id, description):
    tasks = load_tasks()
    for task in tasks:
        if task["id"] == task_id:
            task["description"] = description
            task["updatedAt"] = datetime.now().isoformat()
            save_tasks(tasks)
            logger.info("Task updated: id=%d", task_id)
            print("Task updated successfully")
            return
    logger.warning("Cannot update task: id=%d not found", task_id)
    print("Task not found")


def delete_task(task_id):
    tasks = load_tasks()
    new_tasks = [task for task in tasks if task["id"] != task_id]
    if len(tasks) == len(new_tasks):
        logger.warning("Cannot delete task: id=%d not found", task_id)
        print("Task not found")
    else:
        save_tasks(new_tasks)
        logger.info("Task deleted: id=%d", task_id)
        print("Task deleted successfully")


def mark_status(task_id, status):
    tasks = load_tasks()
    for task in tasks:
        if task["id"] == task_id:
            task["status"] = status
            task["updatedAt"] = datetime.now().isoformat()
            save_tasks(tasks)
            logger.info("Task status changed: id=%d, status=%s", task_id, status)
            print(f"Task marked as {status}")
            return
    logger.warning("Cannot change status: task id=%d not found", task_id)
    print("Task not found")


def list_tasks(filter_status=None):
    tasks = load_tasks()
    if filter_status:
        tasks = [task for task in tasks if task["status"] == filter_status]
    logger.debug("Listing tasks: filter=%s, count=%d", filter_status, len(tasks))

    if not tasks:
        print("No tasks found")
        return

    for task in tasks:
        print(f"[{task['id']}] {task['description']} - {task['status']}")

def parse_task_id(raw_value):
    try:
        return int(raw_value)
    except ValueError:
        logger.warning("Invalid task id %r: an integer is expected", raw_value)
        print(f"Invalid task id: {raw_value}")
        return None

def main():
    setup_logging()
    if len(sys.argv) < 2:
        logger.warning("No command provided")
        print("Usage: task-cli <command> [arguments]")
        return

    command = sys.argv[1]
    logger.debug("Command received: %s, argument_count=%d", command, len(sys.argv) - 2)

    if command == "add":
        if len(sys.argv) < 3:
            logger.warning("Command 'add' called without a description")
            print("Usage: task-cli add <description>")
            return
        add_task(" ".join(sys.argv[2:]))

    elif command == "update":
        if len(sys.argv) < 4:
            logger.warning("Command 'update' called with missing arguments")
            print("Usage: task-cli update <id> <description>")
            return
        task_id = parse_task_id(sys.argv[2])
        if task_id is None:
            return
        update_task(task_id, " ".join(sys.argv[3:]))

    elif command == "delete":
        if len(sys.argv) < 3:
            logger.warning("Command 'delete' called without an id")
            print("Usage: task-cli delete <id>")
            return
        task_id = parse_task_id(sys.argv[2])
        if task_id is None:
            return
        delete_task(task_id)

    elif command == "mark-in-progress":
        if len(sys.argv) < 3:
            logger.warning("Command 'mark-in-progress' called without an id")
            print("Usage: task-cli mark-in-progress <id>")
            return
        task_id = parse_task_id(sys.argv[2])
        if task_id is None:
            return
        mark_status(task_id, "in-progress")

    elif command == "mark-done":
        if len(sys.argv) < 3:
            logger.warning("Command 'mark-done' called without an id")
            print("Usage: task-cli mark-done <id>")
            return
        task_id = parse_task_id(sys.argv[2])
        if task_id is None:
            return
        mark_status(task_id, "done")

    elif command == "list":
        if len(sys.argv) == 2:
            list_tasks()
        else:
            status = sys.argv[2]
            if status not in ["todo", "in-progress", "done"]:
                logger.warning("Invalid status filter: %r", status)
                print("Invalid status. Use: todo, in-progress, or done")
            else:
                list_tasks(status)

    else:
        logger.warning("Unknown command: %r", command)
        print("Unknown command")

if __name__ == "__main__":
    main()