from pydantic import BaseModel, ConfigDict, Field

class TaskSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    id: str
    title: str
    completed: bool


class TaskCreateSchema(BaseModel):
    title: str = Field(min_length=1)


class TaskUpdateSchema(BaseModel):
    title: str | None = Field(default=None, min_length=1)
    completed: bool | None = None
