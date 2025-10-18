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


module_logger = setup_logger()
