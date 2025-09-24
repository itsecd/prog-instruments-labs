from contextlib import suppress
import os
import sqlite3
import logging

from src.config.paths import paths


logger = logging.getLogger(__name__)


def get_connection() -> sqlite3.Connection:
    """
    Gets connection to sqlite3 db

    Returns:
        sqlite3.Connection: connection object
    """
    conn = sqlite3.connect(paths.db_file)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Inits the database with schema if not exists
    """
    if not os.path.exists(paths.db_file):
        with suppress(FileExistsError):
            logger.info("Creating a path to a database file (if not exists): %s", paths.db_file)
            os.makedirs(os.path.dirname(paths.db_file))

        conn = get_connection()
        with open(paths.schema_file, "r") as file:
            conn.executescript(file.read())
            logger.info("A database file created: %s", paths.db_file)

        conn.commit()
        conn.close()
    
    else:
        logger.info("Using a database file: %s", paths.db_file)
