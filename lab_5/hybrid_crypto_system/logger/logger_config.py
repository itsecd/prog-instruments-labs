import logging

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d %(funcName)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    handlers=[logging.FileHandler("logs.log",encoding="utf-8"),
                    logging.StreamHandler()])

logger = logging.getLogger("HybridCryptoSystem")