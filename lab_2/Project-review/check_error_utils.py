import traceback
from tkinter import messagebox
from typing import Optional, Callable, Any
from functools import wraps


def show_error(title: str, message: str, details: Optional[str] = None) -> None:
    """
    Показывает диалоговое окно с сообщением об ошибке.
    
    Args:
        title: Заголовок окна ошибки.
        message: Основной текст сообщения об ошибке.
        details: Дополнительные детали ошибки, добавляются к основному сообщению.
    """
    full_message = message
    if details:
        full_message = f"{message}\n\nДетали:\n{details}"
    messagebox.showerror(title, full_message)


def show_warning(title: str, message: str) -> None:
    """
    Показывает диалоговое окно с предупреждением.
    
    Args:
        title: Заголовок окна предупреждения.
        message: Текст предупреждения.
    """
    messagebox.showwarning(title, message)


def show_info(title: str, message: str) -> None:
    """
    Показывает информационное диалоговое окно.
    
    Args:
        title: Заголовок окна информации.
        message: Информационное сообщение.
    """
    messagebox.showinfo(title, message)


def show_question(title: str, message: str) -> bool:
    """
    Показывает диалог подтверждения с кнопками Да/Нет.
    
    Args:
        title: Заголовок окна вопроса.
        message: Текст вопроса для пользователя.
    
    Returns:
        True если пользователь нажал "Да", False если нажал "Нет".
    """
    return messagebox.askyesno(title, message)


def handle_errors(
    error_title: str = "Ошибка",
    show_traceback: bool = True,
    fallback_return: Any = None,
    status_callback: Optional[Callable] = None
) -> Callable:
    """
    Декоратор для автоматической обработки исключений в функциях.
    Перехватывает любые ошибки, показывает сообщение и возвращает значение по умолчанию.
    
    Args:
        error_title: Заголовок окна ошибки.
        show_traceback: Показывать ли полный отладчик.
        fallback_return: Значение, возвращаемое при ошибке.
        status_callback: Функция для обновления статуса, принимает строку.
    
    Returns:
        Декорированная функция с обработкой ошибок.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                error_message = str(e) if str(e) else f"Ошибка в {func.__name__}"
                details = None
                if show_traceback:
                    details = traceback.format_exc()
                show_error(error_title, error_message, details)
                if status_callback:
                    try:
                        status_callback("Ошибка")
                    except Exception as callback_error:
                        print(f"Ошибка в status_callback: {callback_error}")
                
                return fallback_return
        return wrapper
    return decorator


def confirm_action(title: str, message: str) -> Callable:
    """
    Декоратор для подтверждения действия перед выполнением функции.
    Показывает диалог подтверждения, функция выполняется только при согласии пользователя.
    
    Args:
        title: Заголовок окна подтверждения.
        message: Текст вопроса для пользователя.
    
    Returns:
        Декорированная функция с подтверждением.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if show_question(title, message):
                return func(*args, **kwargs)
            return None
        return wrapper
    return decorator


def safe_file_operation(operation_name: str, status_callback: Optional[Callable] = None) -> Callable:
    """
    Декоратор для безопасных файловых операций.
    
    Args:
        operation_name: Название операции для заголовка ошибки.
        status_callback: Функция для обновления статуса.
    
    Returns:
        Декорированная функция с обработкой ошибок.
    """
    return handle_errors(
        error_title=f"Ошибка при {operation_name}",
        show_traceback=True,
        status_callback=status_callback
    )


def safe_encryption_operation(operation_name: str, status_callback: Optional[Callable] = None) -> Callable:
    """
    Декоратор для безопасных криптографических операций.
    
    Args:
        operation_name: Название операции (не используется).
        status_callback: Функция для обновления статуса.
    
    Returns:
        Декорированная функция с обработкой ошибок.
    """
    return handle_errors(
        error_title="Криптографическая ошибка",
        show_traceback=True,
        status_callback=status_callback
    )