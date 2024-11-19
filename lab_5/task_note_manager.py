from typing import List, Optional, Dict, Union


class TaskNoteManager:
    def __init__(self) -> None:
        """
        Инициализирует менеджер задач и заметок.
        """
        self.tasks: List[Dict[str, Union[str, List[str], bool]]] = []
        self.notes: List[str] = []

    # Методы для управления задачами
    def add_task(self, task: str, tags: Optional[List[str]] = None) -> bool:
        """
        Добавляет новую задачу с опциональными тегами.

        Parameters:
            task (str): Название задачи.
            tags (Optional[List[str]]): Список тегов для задачи.

        Returns:
            bool: True, если задача добавлена, иначе False.
        """
        if task and not any(t["task"] == task for t in self.tasks):
            task_entry = {
                "task": task,
                "tags": tags if tags else [],
                "completed": False,
            }
            self.tasks.append(task_entry)
            return True
        return False

    def remove_task(self, task: str) -> None:
        """
        Удаляет задачу.

        Parameters:
            task (str): Название задачи для удаления.
        """
        self.tasks = [t for t in self.tasks if t["task"] != task]

    def complete_task(self, task: str) -> bool:
        """
        Отмечает задачу как выполненную.

        Parameters:
            task (str): Название задачи для отметки.

        Returns:
            bool: True, если задача найдена и помечена как выполненная, иначе False.
        """
        for t in self.tasks:
            if t["task"] == task:
                t["completed"] = True
                return True
        return False

    def incomplete_task(self, task: str) -> bool:
        """
        Отмечает задачу как невыполненную.

        Parameters:
            task (str): Название задачи для отметки.

        Returns:
            bool: True, если задача найдена и помечена как невыполненная, иначе False.
        """
        for t in self.tasks:
            if t["task"] == task:
                t["completed"] = False
                return True
        return False

    def get_tasks(self) -> List[Dict[str, Union[str, List[str], bool]]]:
        """
        Возвращает список всех задач.

        Returns:
            List[Dict[str, Union[str, List[str], bool]]]: Список задач.
        """
        return self.tasks

    def get_completed_tasks(self) -> List[Dict[str, Union[str, List[str], bool]]]:
        """
        Возвращает список выполненных задач.

        Returns:
            List[Dict[str, Union[str, List[str], bool]]]: Список выполненных задач.
        """
        return [t for t in self.tasks if t["completed"]]

    def get_incomplete_tasks(self) -> List[Dict[str, Union[str, List[str], bool]]]:
        """
        Возвращает список невыполненных задач.

        Returns:
            List[Dict[str, Union[str, List[str], bool]]]: Список невыполненных задач.
        """
        return [t for t in self.tasks if not t["completed"]]

    def get_tasks_by_tag(
        self, tag: str
    ) -> List[Dict[str, Union[str, List[str], bool]]]:
        """
        Возвращает список задач по тегу.

        Parameters:
            tag (str): Тег для фильтрации задач.

        Returns:
            List[Dict[str, Union[str, List[str], bool]]]: Список задач с указанным тегом.
        """
        return [t for t in self.tasks if tag in t["tags"]]

    # Методы для управления заметками
    def add_note(self, note: str) -> bool:
        """
        Добавляет новую заметку.

        Parameters:
            note (str): Текст заметки.

        Returns:
            bool: True, если заметка добавлена, иначе False.
        """
        if note:
            self.notes.append(note)
            return True
        return False

    def remove_note(self, note: str) -> None:
        """
        Удаляет заметку.

        Parameters:
            note (str): Текст заметки для удаления.
        """
        self.notes = [n for n in self.notes if n != note]

    def get_notes(self) -> List[str]:
        """
        Возвращает список всех заметок.

        Returns:
            List[str]: Список заметок.
        """
        return self.notes
