import sys
from loguru import logger


def setup_logger():
    logger.remove()
    logger.add(sys.stdout, 
               level='DEBUG', 
               format='{time:DD.MM.YYYY at HH:mm:ss} | {name} | {level} | {message}'
               )
 