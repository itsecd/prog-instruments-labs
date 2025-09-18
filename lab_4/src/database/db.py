from contextlib import suppress
import os
import sqlite3

from src.config.paths import paths


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(paths.db_file)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    if not os.path.exists(paths.db_file):
        with suppress(FileExistsError):
            os.makedirs(os.path.dirname(paths.db_file))

        conn = get_connection()
        with open(paths.schema_file, "r") as file:
            conn.executescript(file.read())

        conn.commit()
        conn.close()
