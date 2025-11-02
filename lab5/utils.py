import json
import os
from loguru import logger

from cryptography.hazmat.primitives import serialization

def load_settings(settings_path='settings.json'):
    """Loads settings from JSON file with error handling."""
    try:
        if not os.path.exists(settings_path):
            logger.error(f"Settings file not found: {settings_path}")
            raise FileNotFoundError(f"Settings file {settings_path} not found")
        with open(settings_path, 'r') as f:
            settings = json.load(f)
            logger.debug(f"Settings loaded successfully from {settings_path}")
            return settings
    except FileNotFoundError as e:
        logger.error(f"Settings file not found: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON: {e}")
        raise
    except PermissionError as e:
        logger.error(f"File access denied {settings_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading settings: {e}")
        raise

def save_public_key(key, path):
    """Saves public key to file."""
    try:
        with open(path, 'wb') as f:
            key_data = key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            f.write(key_data)
        logger.debug(f"Public key saved to {path}")
    except Exception as e:
        logger.error(f"Error saving public key: {e}")
        raise

def save_private_key(key, path):
    """Saves private key to file."""
    try:
        with open(path, 'wb') as f:
            key_data = key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption()
            )
            f.write(key_data)
        logger.debug(f"Private key saved to {path}")
    except Exception as e:
        logger.error(f"Error saving private key: {e}")
        raise

def serialize_key(key, path, key_type):
    """Serializes key to file if file doesn't exist."""
    if os.path.exists(path):
        logger.warning(f"File {path} already exists, skipping serialization.")
        return

    logger.info(f"Serializing {key_type} key to {path}...")
    output_dir = os.path.dirname(path)
    if output_dir and not os.path.exists(output_dir):
        logger.debug(f"Creating directory: {output_dir}")
        os.makedirs(output_dir)

    match key_type:
        case 'symmetric':
            with open(path, 'wb') as f:
                f.write(key)
            logger.debug(f"Symmetric key serialized to {path}, size: {len(key)} bytes")
        case 'public':
            save_public_key(key, path)
            logger.debug(f"Public key serialized to {path}")
        case 'private':
            save_private_key(key, path)
            logger.debug(f"Private key serialized to {path}")
        case _:
            logger.error(f"Invalid key type: {key_type}")
            raise ValueError(f"Invalid key type: {key_type}")
    
    logger.success(f"Key serialization completed: {path}")