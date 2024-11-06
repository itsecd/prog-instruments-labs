from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload
from uuid import UUID

from core.schemas import ItemCreate, ItemRead, ItemUpdate
from core.models import Item
from database import get_async_session
from auth.models import User


router = APIRouter(
    prefix="/todo",
    tags=["ToDoList"]
)

from main import fastapi_users

current_user = fastapi_users.current_user()

@router.post("/add_item")
async def add_item_to_list(item: ItemCreate, 
                           session: AsyncSession = Depends(get_async_session),
                           user: User = Depends(current_user)):
    item_db = Item(**item.model_dump(), user_id=user.id)
    session.add(item_db)
    await session.commit()

    return {
        "status": "ok",
        "detail": "item added",
        "data": ItemRead(**item.model_dump(), id=item_db.id, is_done=False)
    }


@router.get("/get_items")
async def get_items(sort_by: list[str] = Query(default=["do_till", "1"], max_length=2, min_length=2),
                    user: User = Depends(current_user),
                    session: AsyncSession = Depends(get_async_session)):
    
    if sort_by[0] not in (None, "priority", "do_till"):
        raise HTTPException(status_code=422, detail=
                            {
                            "status": "error",
                            "detail": "item added",
                            "data": "wrong field to sort by"
                            }
        )
    try:
        sort_by[1] = int(sort_by[1])
    except:
        raise HTTPException(status_code=422, detail=
                            {
                            "status": "error",
                            "detail": "item added",
                            "data": "second argument must be integer"
                            }
        )

    query = select(User).options(joinedload(User.to_do_items)).filter(user.id == User.id)
    user_with_items = await session.execute(query)
    user_with_items = user_with_items.unique().scalars().first()

    res = [ItemRead.model_validate(i, from_attributes=True) for i in user_with_items.to_do_items]

    if sort_by[1] != 0:
        res.sort(key=lambda x: getattr(x, sort_by[0]), reverse=True if sort_by[1] > 0 else False)

    return {
        "status": "ok",
        "detail": "got items",
        "data": res
    }


@router.post("/marks_as_done")
async def mark(item_id: UUID, 
               user: User = Depends(current_user),
               session: AsyncSession = Depends(get_async_session)
               ):
    item_db = await session.get(Item, item_id)

    if item_db.user_id != user.id:
        return
    item_db.is_done = True
    await session.commit()

    return {
        "status": "ok",
        "detail": "item is done",
        "data": ItemRead(**item_db.__dict__)
    }


@router.patch("/update_item")
async def update_item(item_id: UUID,
                      new_item: ItemUpdate,
                      session: AsyncSession = Depends(get_async_session),
                      user: User = Depends(current_user)
                      ):
    item_db = await session.get(Item, item_id)
    if item_db.user_id != user.id:
        return
    to_upd = new_item.model_dump()
    for field, val in to_upd.items():
        if val is not None:
            setattr(item_db, field, val)
    await session.commit()

    return {
        "status": "ok",
        "detail": "item updated",
        "data": ItemRead(**item_db.__dict__)
    }


@router.delete("/delete_item")
async def delete_item(item_id,
                      session: AsyncSession = Depends(get_async_session),
                      user: User = Depends(current_user)
                      ):
    item_db = await session.get(Item, item_id)
    if item_db.user_id != user.id:
        return
    await session.delete(item_db)
    await session.commit()

    return {
        "status": "ok",
        "detail": "item deleted",
        "data": None
    }
