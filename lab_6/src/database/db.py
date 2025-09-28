from contextlib import suppress
import os
import sqlite3
import logging

from .base import BaseDB
from src.database.crud import DBMixin


logger = logging.getLogger(__name__)


class DB(BaseDB, DBMixin):
    def __init__(self, file_path: str, schema_file: str):
        """
        Inits the database with schema if not exists
        """
        self.db_file_path = file_path

        if not os.path.exists(self.db_file_path):
            with suppress(FileExistsError):
                logger.info("Creating a path to a database file (if not exists): %s", self.db_file_path)
                os.makedirs(os.path.dirname(self.db_file_path))

            conn = self._get_connection()
            with open(schema_file, "r") as file:
                conn.executescript(file.read())
                logger.info("A database file created: %s", self.db_file_path)

            conn.commit()
            conn.close()
        
        else:
            logger.info("Using a database file: %s", self.db_file_path)

    def _get_connection(self) -> sqlite3.Connection:
        """
        Gets connection to sqlite3 db

        Returns:
            sqlite3.Connection: connection object
        """
        conn = sqlite3.connect(self.db_file_path)
        conn.row_factory = sqlite3.Row
        return conn
