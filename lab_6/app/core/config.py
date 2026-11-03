import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    DATABASE_URL: str
    cors_allowed_origins: list[str]


def get_settings() -> Settings:
    return Settings(
        DATABASE_URL=os.getenv(
            "DATABASE_URL",
            "postgresql+psycopg://postgres:admin@localhost:15432/postgres",
        ),
        cors_allowed_origins = ["http://localhost:3000"]
    )



