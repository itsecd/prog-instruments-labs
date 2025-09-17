from dataclasses import dataclass
from dotenv import load_dotenv
import os


load_dotenv()


@dataclass(frozen=True)
class Config:
    bot_token: str


config = Config(
    bot_token=os.environ["BOT_TOKEN"],
)
