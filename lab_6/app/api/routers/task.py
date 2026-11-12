from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_task_service
from app.schemas.task import TaskCreateSchema, TaskSchema, TaskUpdateSchema
from app.services.task import TaskNotFound, TaskService


router = APIRouter(prefix="/tasks")



@router.get("")
def read_tasks(task_service: TaskService = Depends(get_task_service)) -> list[TaskSchema]:
    return task_service.list_tasks()

@router.post("", status_code=status.HTTP_201_CREATED)
def create_task(payload: TaskCreateSchema, task_service: TaskService = Depends(get_task_service)) -> TaskSchema:
    return task_service.create_task(task_create=payload)


@router.patch("/{task_id}")
def update_task(task_id: str, payload: TaskUpdateSchema, task_service: TaskService = Depends(get_task_service)) -> TaskSchema:
    try:
        return task_service.update_task(task_id=task_id, task_update=payload)
    except TaskNotFound:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")

@router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_task(task_id: str, task_service: TaskService = Depends(get_task_service)) -> None:
    try:
        return task_service.delete_task(task_id=task_id)
    except TaskNotFound:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")