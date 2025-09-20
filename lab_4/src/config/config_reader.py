from dataclasses import dataclass
from dotenv import load_dotenv
import os


load_dotenv()


@dataclass(frozen=True)
class Config:
    """Object for storing config data"""
    bot_token: str


config = Config(
    bot_token=os.environ["BOT_TOKEN"],
)
