import uuid
import json
import logging

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

        extra = {
            "event": "USER_LOGIN",
            "user_email": str(user.email),
            "data": UserRead.model_validate(user, from_attributes=True)
        }

        logging.info(msg="Logged in successfully", extra=extra)

    async def on_after_register(self, user: User, request: Optional[Request] = None):

        content: str = f"<div>Dear {user.username}, " \
        "you has been registred at ToDoList service</div>"
        email: dict[str, str] = get_email_template_dashboard(to=user.email,
                                                            theme="Successful registration",
                                                            content=content)
        send_email_report_dashboard.delay(email)

        extra = {
            "event": "USER_REGISTER",
            "user_email": str(user.email),
            "data": UserRead.model_validate(user, from_attributes=True)
        }

        logging.info(msg="Successfully registered", extra=extra)

    async def on_after_forgot_password(
        self, user: User, token: str, request: Optional[Request] = None
    ):
        content: str = f"<div>Dear {user.username}, use this token to reset your password</div>" \
        f"<div>{token}</div>"
        email: dict[str, str] = get_email_template_dashboard(to=user.email,
                                                            theme="Password reset",
                                                            content=content)
        send_email_report_dashboard.delay(email)

        extra = {
            "event": "USER_REQUESTED_PASSWORD_RESET",
            "user_email": str(user.email),
            "data": token 
        }
        
        logging.info(msg="User requested a password reset", extra=extra)

    async def on_after_reset_password(self, user: User, request: Request | None = None) -> None:
        content: str = f"<div>Dear {user.username}, your password has been reseted</div>"
        email: dict[str, str] = get_email_template_dashboard(to=user.email,
                                                            theme="Successful password reset",
                                                            content=content)
        send_email_report_dashboard.delay(email)

        extra = {
            "event": "USER_PASSWORD_RESET",
            "user_email": str(user.email),
            "data": None
        }
        
        logging.info(msg="Reset user's password", extra=extra)

    async def on_after_request_verify(
        self, user: User, token: str, request: Optional[Request] = None
    ):
        content: str = f"<div>Dear {user.username}, use this token to verify your email</div>" \
        f"<div>{token}</div>"
        email: dict[str, str] = get_email_template_dashboard(to=user.email,
                                                            theme="Email verification",
                                                            content=content)
        send_email_report_dashboard.delay(email)

        extra = {
            "event": "USER_REQUEST_VERIFY",
            "user_email": str(user.email),
            "data": token
        }
        
        logging.info(msg="Requested verification email for user", extra=extra)
    
    async def on_after_verify(self, user: User, request: Request | None = None) -> None:
        content: str = f"<div>Dear {user.username}, your email has been verified"
        email: dict[str, str] = get_email_template_dashboard(to=user.email,
                                                            theme="Successful email verification",
                                                            content=content)
        send_email_report_dashboard.delay(email)

        extra = {
            "event": "USER_VERIFY",
            "user_email": str(user.email),
            "data": None
        }
        
        logging.info(msg="User has been verified", extra=extra)


async def get_user_manager(user_db=Depends(get_user_db)):
    yield UserManager(user_db)
