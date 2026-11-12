from pydantic import BaseModel, ConfigDict, Field


class CategorySchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    name: str


class CategoryCreateSchema(BaseModel):
    name: str = Field(min_length=1)


class CategoryUpdateSchema(BaseModel):
    name: str | None = Field(default=None, min_length=1)