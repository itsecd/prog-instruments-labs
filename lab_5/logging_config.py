import logging
import logging.config
import os
from datetime import datetime


def setup_api_logging():
    """
    Basic logging configuration for Django API.
    """
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'simple': {
                'format': '%(levelname)s - %(message)s'
            }
        },
        'handlers': {
            'console': {
                'level': 'INFO',
                'class': 'logging.StreamHandler',
                'formatter': 'simple'
            },
            'api_file': {
                'level': 'INFO',
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': os.path.join(log_dir, 'api.log'),
                'maxBytes': 1024 * 1024 * 5,  # 5 MB
                'backupCount': 5,
                'formatter': 'detailed',
                'encoding': 'utf-8'
            }
        },
        'loggers': {
            'users_api': {
                'handlers': ['console', 'api_file'],
                'level': 'INFO',
                'propagate': False,
            }
        }
    }

    logging.config.dictConfig(logging_config)


# Basic logger instance
api_logger = logging.getLogger('users_api')