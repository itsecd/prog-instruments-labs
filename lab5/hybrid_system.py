import json
from asymmetric_crypt import decrypt_symmetric_key, encrypt_symmetric_key, generate_asymmetric_keys
from symmetric_crypt import decrypt_file, encrypt_file, generate_symmetric_key
from utils import serialize_key
from loguru import logger


def load_settings(config_file: str) -> dict:
    """
    Loads settings from JSON file.
    """
    try:
        with open(config_file, 'r') as f:
            settings = json.load(f)
            logger.debug(f"Settings loaded successfully from {config_file}")
            return settings
    except FileNotFoundError:
        logger.error(f"Settings file '{config_file}' not found.")
        raise
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in file '{config_file}'.")
        raise
    except Exception as e:
        logger.error(f"An error occurred while loading settings: {e}")
        raise


def validate_settings(settings: dict, required_keys: list) -> None:
    """
    Validates required keys in settings.
    """
    missing_keys = [key for key in required_keys if key not in settings]
    if missing_keys:
        logger.error(f"The required keys are missing: {', '.join(missing_keys)}")
        raise KeyError(f"The required keys are missing: {', '.join(missing_keys)}")


def generate_keys(settings: dict) -> None:
    """
    Generates and saves keys specified in settings.
    """
    try:
        required_keys = settings.get('required_keys', {}).get('generate', [])
        validate_settings(settings, required_keys)
        logger.info("Starting key generation process")

        sym_key = generate_symmetric_key()
        private_key, public_key = generate_asymmetric_keys()

        serialize_key(public_key, settings[required_keys[0]], 'public')  # public_key
        serialize_key(private_key, settings[required_keys[1]], 'private')  # secret_key
        encrypt_symmetric_key(sym_key, public_key, settings[required_keys[2]])  # symmetric_key
        
        logger.success("Key generation completed successfully")

    except KeyError as e:
        logger.error(f"Key error: {e}")
        raise
    except Exception as e:
        logger.error(f"An error occurred during key generation: {e}")
        raise


def encrypt_data(settings: dict) -> None:
    """
    Encrypts data.
    """
    try:
        required_keys = settings.get('required_keys', {}).get('encrypt', [])
        validate_settings(settings, required_keys)
        logger.info("Starting data encryption process")

        sym_key = decrypt_symmetric_key(
            settings[required_keys[0]],  
            settings[required_keys[1]]   
        )
        encrypt_file(
            settings[required_keys[2]],
            settings[required_keys[3]],
            sym_key
        )
        logger.success("Data encryption completed successfully")

    except KeyError as e:
        logger.error(f"Key error: {e}")
        raise
    except FileNotFoundError:
        logger.error("One of the specified files could not be found.")
        raise
    except Exception as e:
        logger.error(f"An error occurred while encrypting data: {e}")
        raise


def decrypt_data(settings: dict) -> None:
    """
    Decrypts data.
    """
    try:
        required_keys = settings.get('required_keys', {}).get('decrypt', [])
        validate_settings(settings, required_keys)
        logger.info("Starting data decryption process")

        sym_key = decrypt_symmetric_key(
            settings[required_keys[0]], 
            settings[required_keys[1]]  
        )
        decrypt_file(
            settings[required_keys[2]],  
            settings[required_keys[3]], 
            sym_key
        )
        logger.success("Data decryption completed successfully")

    except KeyError as e:
        logger.error(f"Key error: {e}")
        raise
    except FileNotFoundError:
        logger.error("One of the specified files could not be found.")
        raise
    except Exception as e:
        logger.error(f"An error occurred while decrypting data: {e}")
        raise