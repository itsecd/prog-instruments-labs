from dataclasses import dataclass
import os


CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

SRC_DIR = os.path.dirname(CONFIG_DIR)

ROOT_DIR = os.path.dirname(SRC_DIR)

DATA_DIR = os.path.join(ROOT_DIR, "data")

DB_DIR = os.path.join(DATA_DIR, "database.db")


@dataclass(frozen=True)
class Paths:
    config_dir: str
    data_dir: str
    db_dir: str


paths = Paths(
    config_dir=CONFIG_DIR,
    data_dir=DATA_DIR,
    db_dir=DB_DIR,
)
