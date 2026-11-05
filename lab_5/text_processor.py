import logging

logger = logging.getLogger(__name__)


def save_text(text: str, path: str):
    """
    Save data to file
    :param text: Source bytes of data
    :param path: Path to save bytes
    """
    logger.debug("Saving data to %s: %s bytes", path, len(text))
    try:
        with open(path, "wb") as file:
            file.write(text)
        logger.debug("Data saved successfully: %s", path)
    except Exception:
        logger.exception("Failed to save data to %s", path)
        raise


def read_text(path: str):
    """
    Read data from file.
    :param path: Path to text file
    :return: Bytes of data
    """
    logger.debug("Reading data from %s", path)
    try:
        with open(path, "rb") as file:
            data = file.read()
        logger.debug("Data read from %s: %s bytes", path, len(data))
        return data
    except Exception:
        logger.exception("Failed to read data from %s", path)
        raise


def read_chipher(path: str):
    """
    Read chipher text from file.
    :param path: Path to file with chipertext
    :return: iv sequence and chiphertext
    """
    logger.debug("Reading ciphertext from %s", path)
    try:
        with open(path, "rb") as f:
            iv = f.read(16)
            c_text = f.read()
        logger.debug(
            "Ciphertext read: file=%s, iv_size=%s, ciphertext_size=%s",
            path,
            len(iv),
            len(c_text),
        )
        return iv, c_text
    except Exception:
        logger.exception("Failed to read ciphertext from %s", path)
        raise
