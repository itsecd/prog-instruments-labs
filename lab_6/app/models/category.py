from sqlalchemy.orm import Mapped
from .base import Base

class CategoryORM(Base):
    __tablename__ = "categories"
    name: Mapped[str]