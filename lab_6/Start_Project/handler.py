import sqlite3
import logging
from datetime import datetime


class SQLiteHandler(logging.Handler):
    """Хендлер для записи логов в базу данных SQLite."""

    def __init__(self, db_path: str):
        super().__init__()
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Создает таблицу, если ее еще нет."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    level TEXT,
                    logger_name TEXT,
                    message TEXT
                )
            """)
            conn.commit()

    def emit(self, record: logging.LogRecord):
        """
        Вызывается автоматически при каждом вызове logger.info() и т.д.
        Записывает данные в БД, разбивая их по полям.
        """
        try:
            timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO logs (timestamp, level, logger_name, message) "
                    "VALUES (?, ?, ?, ?)",
                    (timestamp, record.levelname, record.name, record.getMessage())
                )
                conn.commit()
        except Exception:
            self.handleError(record)