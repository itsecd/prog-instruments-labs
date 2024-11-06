import uuid

from sqlalchemy import (String, 
                        Integer, 
                        DATE, 
                        TIMESTAMP, 
                        UUID, 
                        Boolean, 
                        ForeignKey)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base


class Item(Base):
    __tablename__ = 'item'
    id: Mapped[uuid.UUID] = mapped_column(UUID, primary_key=True, default=lambda: uuid.uuid4())
    user_id: Mapped[uuid.UUID] = mapped_column(UUID, ForeignKey("user.id"))
    name: Mapped[str] = mapped_column(String)
    comment: Mapped[str] = mapped_column(String, default=None, nullable=True)
    priority: Mapped[int] = mapped_column(Integer, default=0, nullable=True)
    do_till: Mapped[DATE | TIMESTAMP] = mapped_column(TIMESTAMP(timezone=True), default=None, nullable=True)
    is_done: Mapped[bool] = mapped_column(Boolean, default=False)
    user = relationship("User", back_populates="to_do_items")
