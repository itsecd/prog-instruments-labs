from dataclasses import dataclass
import os


CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

SRC_DIR = os.path.dirname(CONFIG_DIR)

ROOT_DIR = os.path.dirname(SRC_DIR)

DATA_DIR = os.path.join(ROOT_DIR, "data")

DATABASE_DIR = os.path.join(SRC_DIR, "database")

SCHEMA_FILE = os.path.join(DATABASE_DIR, "schema.sql")

DB_FILE = os.path.join(DATA_DIR, "database.db")


@dataclass(frozen=True)
class Paths:
    """Object for storing paths"""
    config_dir: str
    data_dir: str
    db_file: str
    schema_file: str


paths = Paths(
    config_dir=CONFIG_DIR,
    data_dir=DATA_DIR,
    db_file=DB_FILE,
    schema_file=SCHEMA_FILE,
)
