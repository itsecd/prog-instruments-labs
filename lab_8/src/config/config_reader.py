from dataclasses import dataclass
from dotenv import load_dotenv
import os


load_dotenv()


@dataclass(frozen=True)
class Config:
    """Object for storing config data"""
    tg_bot_token: str


config = Config(
    tg_bot_token=os.environ["TG_BOT_TOKEN"],
)
