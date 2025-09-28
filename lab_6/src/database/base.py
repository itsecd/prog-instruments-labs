from abc import ABC, abstractmethod
from typing import Any

class BaseDB:
    pass


class BaseDBMixin(ABC):
    @abstractmethod
    def create_user(self, tg_id: int, tasks_reversed: bool = False): pass
    
    @abstractmethod
    def set_tasks_reversed(self, tg_id: int, value: bool): pass

    @abstractmethod
    def get_user(self, tg_id: int) -> dict[str, Any] | None: pass

    @abstractmethod
    def add_task(self, tg_id: int, task: str): pass

    @abstractmethod
    def get_task_by_idx(self, tg_id: int, idx: int) -> dict[str, Any] | None: pass

    @abstractmethod
    def update_task_by_idx(self, tg_id: int, idx: int, new_task: str): pass

    @abstractmethod
    def delete_task_by_idx(self, tg_id: int, idx: int): pass

    @abstractmethod
    def get_tasks_count(self, tg_id: int) -> int: pass

    @abstractmethod
    def get_tasks(self, tg_id: int) -> list[dict[str, Any]]: pass