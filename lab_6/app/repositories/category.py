from sqlalchemy.orm import Session
from app.models.category import CategoryORM
from sqlalchemy import select


class CategoryRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_all(self) -> list[CategoryORM]:
        return list(self.db.scalars(select(CategoryORM)).all())

    def get_by_id(self, category_id: str) -> CategoryORM | None:
        return self.db.get(CategoryORM, category_id)

    def create(self, name) -> CategoryORM:
        category = CategoryORM(name=name)
        self.db.add(category)
        return category

    def delete(self, CategoryORM) -> None:
        self.db.delete(CategoryORM)