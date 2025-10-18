import functools
import loguru



def setup_logger():

    loguru.logger.remove()

    file_format = (
                    "{time:YYYY-MM-DD HH:mm:ss} | "
                    "{level: <8} | "
                    "{name}:{function}:{line} | "
                    "extra={extra} | "
                    "{message}"
                   )

    loguru.logger.add(
        "logs/logs.log",
        format=file_format,
        level="INFO",
        rotation="1 day",
        retention="60 days"
    )

    return loguru.logger


def log_errors(logger):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"Ошибка в {func.__name__}",
                    error=str(e),
                    args=args,
                    kwargs=kwargs,
                    exc_info=True
                )
                raise
        return wrapper
    return decorator


module_logger = setup_logger()
