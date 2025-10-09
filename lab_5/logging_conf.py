import logging
import logging.config
import sys

logging. config. fileConfig('logging.conf')
logger = logging.getLogger ('appLogger')