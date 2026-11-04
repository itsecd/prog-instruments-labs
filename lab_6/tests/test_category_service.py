"""Тесты сервиса категорий с подменой репозитория и сессии на моки."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.repositories.category import CategoryRepository
from app.schemas.category import CategorySchema, CategoryCreateSchema, CategoryUpdateSchema
from app.services.category import CategoryService, CategoryNotFound


@pytest.fixture
def db() -> Mock:
    """Мок сессии БД"""
    return Mock()


@pytest.fixture
def service(db: Mock) -> CategoryService:
    """Сервис категорий с подменой репозитория на Mock"""
    category_service = CategoryService(db)
    category_service.category_repository = Mock(spec=CategoryRepository)
    return category_service


def make_category(name: str = "Category 1") -> SimpleNamespace:
    """Создаёт объект имитирующий категорию из БД"""
    return SimpleNamespace(id="cat-1", name=name)


def test_list_categories_converts_orm_objects(service: CategoryService) -> None:
    """Список категорий из БД превращается в схемы API"""
    service.category_repository.get_all.return_value = [
        SimpleNamespace(id="cat-1", name="Category 1"),
        SimpleNamespace(id="cat-2", name="Category 2"),
    ]
    result = service.list_categories()
    assert result == [
        CategorySchema(id="cat-1", name="Category 1"),
        CategorySchema(id="cat-2", name="Category 2"),
    ]


def test_create_category_saves_and_commits(
    service: CategoryService,
    db: Mock
) -> None:
    """Тест на создание категории"""
    service.category_repository.create.return_value = make_category()
    result = service.create_category(CategoryCreateSchema(name="Category 1"))
    service.category_repository.create.assert_called_once_with(name="Category 1")
    db.commit.assert_called_once()
    assert result == CategorySchema(id="cat-1", name="Category 1")


def test_update_missing_category_raises_and_not_commit(
    service: CategoryService,
    db: Mock
) -> None:
    """Тест на обновление несуществующей категории"""
    service.category_repository.get_by_id.return_value = None
    with pytest.raises(CategoryNotFound):
        service.update_category("missing", CategoryUpdateSchema(name="New name"))
    db.commit.assert_not_called()


@pytest.mark.parametrize(
    ("new_name", "expected_name"),
    [("Category 2", "Category 2"), (None, "Category 1")],
    ids=["new name", "no name"],
)
def test_update_category_changes_name_only_when_given(
    service: CategoryService,
    db: Mock,
    new_name: str | None,
    expected_name: str,
) -> None:
    """Параметризованный тест"""
    category = make_category()
    service.category_repository.get_by_id.return_value = category
    result = service.update_category(
        "cat-1", CategoryUpdateSchema(name=new_name)
    )
    assert category.name == expected_name
    assert result == CategorySchema(id="cat-1", name=expected_name)
    db.commit.assert_called_once()


def test_delete_category_removes_found_category(
    service: CategoryService,
    db: Mock
) -> None:
    """Тест на удаление категории по id единожды"""
    category = make_category()
    service.category_repository.get_by_id.return_value = category
    service.delete_category("cat-1")
    service.category_repository.delete.assert_called_once_with(category)
    db.commit.assert_called_once()


def test_delete_missing_category_raises_and_not_delete(
    service: CategoryService,
    db: Mock
) -> None:
    """Тест на удаление несуществующей категории"""
    service.category_repository.get_by_id.return_value = None
    with pytest.raises(CategoryNotFound):
        service.delete_category("missing")
    service.category_repository.delete.assert_not_called()
    db.commit.assert_not_called()