import pytest
from unittest.mock import patch
from gui import App


@pytest.fixture
def app():
    """Создает и возвращает экземпляр приложения App для использования в тестах."""
    return App()


def test_add_task(app):
    """
    Тестирует функцию добавления задачи в приложение.

    Проверяет, что метод askstring для ввода задачи возвращает корректное значение,
    и что после добавления задачи вызывается сообщение об успешном добавлении.
    """
    with patch("tkinter.simpledialog.askstring", return_value="Test Task"):
        with patch("tkinter.messagebox.showinfo") as mock_showinfo:
            app.add_task()  # Вызов метода для добавления задачи
            # Проверка, что showinfo был вызван с ожидаемыми аргументами
            mock_showinfo.assert_called_once_with("Success", "Task added successfully!")


def test_remove_task(app):
    """
    Тестирует функцию удаления задачи из приложения.

    Сначала добавляет задачу, затем проверяет, что задача удаляется корректно,
    и что после удаления вызывается сообщение об успешном удалении.
    """
    app.manager.add_task("Test Task", [])  # Добавление задачи перед тестом
    with patch("tkinter.simpledialog.askstring", return_value="Test Task"):
        with patch("tkinter.messagebox.showinfo") as mock_showinfo:
            app.remove_task()  # Вызов метода для удаления задачи
            # Проверка, что showinfo был вызван с ожидаемыми аргументами
            mock_showinfo.assert_called_once_with(
                "Success", "Task removed successfully!"
            )


def test_complete_task(app):
    """
    Тестирует функцию завершения задачи в приложении.

    Сначала добавляет задачу, затем проверяет, что задача завершается корректно,
    и что после завершения вызывается сообщение об успешном завершении.
    """
    app.manager.add_task("Test Task", [])  # Добавление задачи перед тестом
    with patch("tkinter.simpledialog.askstring", return_value="Test Task"):
        with patch("tkinter.messagebox.showinfo") as mock_showinfo:
            app.complete_task()  # Вызов метода для завершения задачи
            # Проверка, что showinfo был вызван с ожидаемыми аргументами
            mock_showinfo.assert_called_once_with(
                "Success", "Task marked as completed!"
            )


def test_show_tasks(app):
    """
    Тестирует функцию отображения задач в приложении.

    Сначала добавляет задачу, затем проверяет, что при отображении задач
    вызывается сообщение, подтверждающее успешный вывод задач.
    """
    app.manager.add_task("Test Task", [])  # Добавление задачи перед тестом
    with patch("tkinter.messagebox.showinfo") as mock_showinfo:
        app.show_tasks()  # Вызов метода для отображения задач
        # Проверка, что showinfo был вызван хотя бы один раз
        mock_showinfo.assert_called_once()
