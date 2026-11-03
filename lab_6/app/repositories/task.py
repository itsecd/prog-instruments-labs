from sqlalchemy.orm import Session
from sqlalchemy import select

from app.models.task import TaskORM


class TaskRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_all(self) -> list[TaskORM]:
        return list(self.db.scalars(select(TaskORM)).all())

    def get_by_id(self, task_id: str) -> TaskORM | None:
        return self.db.get(TaskORM, task_id)

    def create(self, title) -> TaskORM:
        task = TaskORM(title=title, completed=False)
        self.db.add(task)
        return task

    def delete(self, TaskORM) -> None:
        self.db.delete(TaskORM)