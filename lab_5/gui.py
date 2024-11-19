import tkinter as tk
from tkinter import messagebox, simpledialog

from task_note_manager import TaskNoteManager


class App(tk.Tk):
    def __init__(self) -> None:
        """Инициализирует приложение."""
        super().__init__()
        self.manager = TaskNoteManager()
        self.title("Task and Note Manager")
        self.geometry("400x400")

        self.create_widgets()

    def create_widgets(self) -> None:
        """Создает виджеты интерфейса."""
        # Заголовок
        title_label = tk.Label(self, text="Task and Note Manager", font=("Arial", 16))
        title_label.pack(pady=10)

        # Кнопки
        task_button = tk.Button(self, text="Add Task", command=self.add_task)
        task_button.pack(pady=5)

        remove_task_button = tk.Button(
            self, text="Remove Task", command=self.remove_task
        )
        remove_task_button.pack(pady=5)

        complete_task_button = tk.Button(
            self, text="Complete Task", command=self.complete_task
        )
        complete_task_button.pack(pady=5)

        incomplete_task_button = tk.Button(
            self, text="Mark Incomplete Task", command=self.mark_incomplete_task
        )
        incomplete_task_button.pack(pady=5)

        note_button = tk.Button(self, text="Add Note", command=self.add_note)
        note_button.pack(pady=5)

        remove_note_button = tk.Button(
            self, text="Remove Note", command=self.remove_note
        )
        remove_note_button.pack(pady=5)

        show_tasks_button = tk.Button(self, text="Show Tasks", command=self.show_tasks)
        show_tasks_button.pack(pady=5)

        show_notes_button = tk.Button(self, text="Show Notes", command=self.show_notes)
        show_notes_button.pack(pady=5)

    def add_task(self) -> None:
        """Добавляет новую задачу через диалоговое окно."""
        task = simpledialog.askstring("Input", "Enter task:")
        if task:
            tags = simpledialog.askstring("Input", "Enter tags (comma separated):")
            tags_list = tags.split(",") if tags else []
            if self.manager.add_task(task, tags_list):
                messagebox.showinfo("Success", "Task added successfully!")
            else:
                messagebox.showwarning("Warning", "Task already exists!")

    def remove_task(self) -> None:
        """Удаляет задачу через диалоговое окно."""
        task = simpledialog.askstring("Input", "Enter task to remove:")
        if task:
            self.manager.remove_task(task)
            messagebox.showinfo("Success", "Task removed successfully!")

    def complete_task(self) -> None:
        """Отмечает задачу как выполненную через диалоговое окно."""
        task = simpledialog.askstring("Input", "Enter task to complete:")
        if task:
            if self.manager.complete_task(task):
                messagebox.showinfo("Success", "Task marked as completed!")
            else:
                messagebox.showwarning("Warning", "Task not found!")

    def mark_incomplete_task(self) -> None:
        """Отмечает задачу как невыполненную через диалоговое окно."""
        task = simpledialog.askstring("Input", "Enter task to mark as incomplete:")
        if task:
            if self.manager.incomplete_task(task):
                messagebox.showinfo("Success", "Task marked as incomplete!")
            else:
                messagebox.showwarning("Warning", "Task not found!")

    def add_note(self) -> None:
        """Добавляет новую заметку через диалоговое окно."""
        note = simpledialog.askstring("Input", "Enter note:")
        if note:
            if self.manager.add_note(note):
                messagebox.showinfo("Success", "Note added successfully!")
            else:
                messagebox.showwarning("Warning", "Note cannot be empty!")

    def remove_note(self) -> None:
        """Удаляет заметку через диалоговое окно."""
        note = simpledialog.askstring("Input", "Enter note to remove:")
        if note:
            self.manager.remove_note(note)
            messagebox.showinfo("Success", "Note removed successfully!")

    def show_tasks(self) -> None:
        """Отображает список задач в диалоговом окне."""
        tasks = self.manager.get_tasks()
        tasks_list = "\n".join(
            [
                f"{t['task']} - {'Completed' if t['completed'] else 'Incomplete'}"
                for t in tasks
            ]
        )
        messagebox.showinfo("Tasks", tasks_list if tasks else "No tasks available.")

    def show_notes(self) -> None:
        """Отображает список заметок в диалоговом окне."""
        notes = self.manager.get_notes()
        notes_list = "\n".join(notes)
        messagebox.showinfo("Notes", notes_list if notes else "No notes available.")


if __name__ == "__main__":
    app = App()
    app.mainloop()
