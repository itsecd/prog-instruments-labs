from fastapi_users_db_sqlalchemy import SQLAlchemyBaseUserTableUUID
from sqlalchemy import String
from sqlalchemy.orm import mapped_column, Mapped, relationship

from database import Base


class User(SQLAlchemyBaseUserTableUUID, Base):
    __tablename__ = "user"
    username: Mapped[str] = mapped_column(String, nullable=False)
    redis_token_key: Mapped[str] = mapped_column(String, default=None, nullable=True)
    to_do_items = relationship("Item", back_populates="user")
    