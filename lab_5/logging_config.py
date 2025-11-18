import logging
import logging.config
import os
from datetime import datetime


def setup_api_logging():
    """
    Enhanced logging configuration with separate loggers for different components.
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
            },
            'auth_format': {
                'format': '%(asctime)s - AUTH - %(levelname)s - %(message)s - User: %(user)s'
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
            },
            'auth_file': {
                'level': 'INFO',
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': os.path.join(log_dir, 'auth.log'),
                'maxBytes': 1024 * 1024 * 5,
                'backupCount': 5,
                'formatter': 'detailed',
                'encoding': 'utf-8'
            },
            'error_file': {
                'level': 'ERROR',
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': os.path.join(log_dir, 'errors.log'),
                'maxBytes': 1024 * 1024 * 5,
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
            },
            'auth': {
                'handlers': ['console', 'auth_file'],
                'level': 'INFO',
                'propagate': False,
            },
            'profile': {
                'handlers': ['console', 'api_file'],
                'level': 'INFO',
                'propagate': False,
            },
            'files': {
                'handlers': ['console', 'api_file'],
                'level': 'INFO',
                'propagate': False,
            }
        },
        'root': {
            'handlers': ['console', 'error_file'],
            'level': 'ERROR',
        }
    }

    logging.config.dictConfig(logging_config)


# Separate logger instances for different components
api_logger = logging.getLogger('users_api')
auth_logger = logging.getLogger('auth')
profile_logger = logging.getLogger('profile')
files_logger = logging.getLogger('files')