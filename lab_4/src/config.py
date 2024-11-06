import dotenv, os


dotenv.load_dotenv()
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = os.getenv("REDIS_PORT")

RESET_PASSWORD_TOKEN_SECRET = os.getenv("RESET_PASSWORD_TOKEN_SECRET")
VERIFICATION_TOKEN_SECRET = os.getenv("VERIFICATION_TOKEN_SECRET")

SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SMTP_USER = os.getenv("SMTP_USER")
