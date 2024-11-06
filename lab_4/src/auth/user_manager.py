import uuid
import json

from typing import Optional

from fastapi import Depends, Request, Response
from fastapi_users import BaseUserManager, UUIDIDMixin

from auth.models import User
from auth.schemas import UserRead
from auth.utils import get_user_db
from auth.auth_backend import redis
from config import RESET_PASSWORD_TOKEN_SECRET, VERIFICATION_TOKEN_SECRET
from tasks.emails import get_email_template_dashboard, send_email_report_dashboard


class UserManager(UUIDIDMixin, BaseUserManager[User, uuid.UUID]):
    reset_password_token_secret = RESET_PASSWORD_TOKEN_SECRET
    verification_token_secret = VERIFICATION_TOKEN_SECRET

    async def on_after_login(self, user: User, 
                             request: Request | None = None, response: Response | None = None
                             ) -> None:
        response_body = response.body.decode('utf-8')
        response_body = json.loads(response_body)
        
        if user.redis_token_key is not None:
            await redis.delete(user.redis_token_key)
        key = f"fastapi_users_token:{response_body['access_token']}"
        await self._update(user, {"redis_token_key": key})

        print(f"user {user.id} has logined")

        return {
        "status": "ok",
        "detail": "logged in",
        "data": UserRead.model_validate(user, from_attributes=True)
        }

    async def on_after_register(self, user: User, request: Optional[Request] = None):

        content: str = f"<div>Dear {user.username}, " \
        "you has been registred at ToDoList service</div>"
        email: dict[str, str] = get_email_template_dashboard(to=user.email,
                                                            theme="Successful registration",
                                                            content=content)
        send_email_report_dashboard.delay(email)

        print(f"User {user.id} has registered.")

        return {
        "status": "ok",
        "detail": "you have been registred",
        "data": UserRead.model_validate(user, from_attributes=True)
        }

    async def on_after_forgot_password(
        self, user: User, token: str, request: Optional[Request] = None
    ):
        content: str = f"<div>Dear {user.username}, use this token to reset your password</div>" \
        f"<div>{token}</div>"
        email: dict[str, str] = get_email_template_dashboard(to=user.email,
                                                            theme="Password reset",
                                                            content=content)
        send_email_report_dashboard.delay(email)

        print(f"User {user.id} has forgot their password. Reset token: {token}")

        return {
        "status": "ok",
        "detail": "requested password reset",
        "data": token
        }

    async def on_after_reset_password(self, user: User, request: Request | None = None) -> None:
        content: str = f"<div>Dear {user.username}, your password has been reseted</div>"
        email: dict[str, str] = get_email_template_dashboard(to=user.email,
                                                            theme="Successful password reset",
                                                            content=content)
        send_email_report_dashboard.delay(email)

        print(f"User {user.id}has reseted password")

        return {
        "status": "ok",
        "detail": "password reseted",
        "data": None
        }

    async def on_after_request_verify(
        self, user: User, token: str, request: Optional[Request] = None
    ):
        content: str = f"<div>Dear {user.username}, use this token to verify your email</div>" \
        f"<div>{token}</div>"
        email: dict[str, str] = get_email_template_dashboard(to=user.email,
                                                            theme="Email verification",
                                                            content=content)
        send_email_report_dashboard.delay(email)

        print(f"Verification requested for user {user.id}. Verification token: {token}")

        return {
        "status": "ok",
        "detail": "requested verification",
        "data": token
        }
    
    async def on_after_verify(self, user: User, request: Request | None = None) -> None:
        content: str = f"<div>Dear {user.username}, your email has been verified"
        email: dict[str, str] = get_email_template_dashboard(to=user.email,
                                                            theme="Successful email verification",
                                                            content=content)
        send_email_report_dashboard.delay(email)

        print(f"User {user.id} has been verified")

        return {
        "status": "ok",
        "detail": "email verified",
        "data": None
        }


async def get_user_manager(user_db=Depends(get_user_db)):
    yield UserManager(user_db)
