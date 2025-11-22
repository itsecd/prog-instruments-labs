import logging
import logging.config
import os
import time
from functools import wraps


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
            'performance': {
                'format': '%(asctime)s - PERFORMANCE - %(levelname)s - %(message)s - Time: %(execution_time).3fs'
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
            },
            'performance_file': {
                'level': 'INFO',
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': os.path.join(log_dir, 'performance.log'),
                'maxBytes': 1024 * 1024 * 5,
                'backupCount': 5,
                'formatter': 'performance',
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
            },
            'performance': {
                'handlers': ['performance_file'],
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
performance_logger = logging.getLogger('performance')


def log_execution_time(logger_name='performance'):
    """
    Decorator to log method execution time and performance.

    Args:
        logger_name: Name of the logger to use for performance logging
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(logger_name)
            start_time = time.time()

            # Extract request info for API views
            request = None
            for arg in args:
                if hasattr(arg, 'user') and hasattr(arg, 'method'):
                    request = arg
                    break

            user_info = ""
            if request and hasattr(request, 'user'):
                user_info = f"User: {getattr(request.user, 'username', 'Anonymous')}"

            logger.info(f"Starting {func.__name__} - {user_info}")

            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time

                # Log performance based on execution time
                if execution_time > 1.0:
                    logger.warning(
                        f"Slow operation: {func.__name__} took {execution_time:.3f}s - {user_info}",
                        extra={'execution_time': execution_time}
                    )
                elif execution_time > 0.5:
                    logger.info(
                        f"Moderate operation: {func.__name__} took {execution_time:.3f}s - {user_info}",
                        extra={'execution_time': execution_time}
                    )
                else:
                    logger.debug(
                        f"Fast operation: {func.__name__} took {execution_time:.3f}s - {user_info}",
                        extra={'execution_time': execution_time}
                    )

                return result

            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    f"Error in {func.__name__} after {execution_time:.3f}s: {str(e)} - {user_info}",
                    extra={'execution_time': execution_time},
                    exc_info=True
                )
                raise

        return wrapper

    return decorator


class PerformanceMixin:
    """
    Mixin class to add performance logging to class-based views.
    """

    @property
    def performance_logger(self):
        return logging.getLogger('performance')

    def log_performance(self, operation, execution_time):
        """Log performance metrics."""
        if execution_time > 1.0:
            self.performance_logger.warning(
                f"Slow {operation}: {execution_time:.3f}s",
                extra={'execution_time': execution_time}
            )
        else:
            self.performance_logger.info(
                f"{operation}: {execution_time:.3f}s",
                extra={'execution_time': execution_time}
            )

    def retrieve(self, request, param, param1):
        pass

    def update(self, request, param, param1):
        pass
