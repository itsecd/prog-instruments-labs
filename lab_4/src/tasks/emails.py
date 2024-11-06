import smtplib
import sys, os

from email.message import EmailMessage
from celery import Celery

sys.path.append(os.path.join(sys.path[0], 'src'))

from config import SMTP_PASSWORD, SMTP_USER, REDIS_HOST, REDIS_PORT


SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465

celery = Celery('tasks', broker=f'redis://{REDIS_HOST}:{REDIS_PORT}')

def get_email_template_dashboard(to: str, theme: str, content: str) -> dict[str, str]:
    email = dict()
    email['Subject'] = theme
    email['From'] = SMTP_USER
    email['To'] = to

    email['Content'] = content
    return email


@celery.task(name='tasks.emails.send_email_report_dashboard')
def send_email_report_dashboard(dict_email: dict[str, str]):
    email = EmailMessage()
    email['Subject'] = dict_email['Subject']
    email['From'] = dict_email['From']
    email['To'] = dict_email['To']
    email.set_content(dict_email['Content'], subtype='html')
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as server:
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.send_message(email)

#celery -A src.tasks.emails:celery worker --loglevel=INFO --pool=solo