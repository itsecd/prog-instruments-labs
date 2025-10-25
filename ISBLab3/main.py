from crypto_system_app import Cryptosys
from loguru import logger as log

if __name__ == "__main__":
    log.info("init program")
    app = Cryptosys()
    app.console_app()
