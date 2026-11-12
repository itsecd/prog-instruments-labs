from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import get_settings


settings = get_settings()
engine = create_engine(settings.DATABASE_URL)
Sessionlocal = sessionmaker(bind=engine)


def get_db():
    """Функция для инъекции базы данных"""
    db = Sessionlocal()

    try:
        yield db
    finally:
        db.close()