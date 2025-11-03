import argparse
import logging


logger = logging.getLogger("parser")


def get_parse() -> tuple:
    """
    Parsing arguments
    :return: tuple of pathes
    """
    logger.info("FUNCTION_STARTED - Parsing command line arguments")

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('img_path', type = str, help = 'Path to image')
        parser.add_argument('save_path', type = str, help = 'Path to file')
        args = parser.parse_args()

        logger.info("ARGUMENTS_PARSED - Image path: %s, Save path: %s", 
            args.img_path, 
            args.save_path)
        
        return args.img_path, args.save_path
    except Exception as e:
        logger.error("ARGUMENT_PARSING_FAILED - Error: %s", str(e))
        raise SyntaxError("Invalid data")
    

def validate_paths(img_path: str, save_path: str) -> bool:
    """
    Check paths are correct or not.
    :param img_path: path to image
    :param save_path: path to save
    :return: True if paths are valide
    """
    logger.debug(
        "VALIDATING_PATHS - Input: %s, Output: %s", 
        img_path, 
        save_path)
    
    if not img_path or not save_path:
        logger.error("EMPTY_PATHS - One or both paths are empty")
        raise ValueError("Paths cannot be empty")
    
    if not img_path.lower().endswith( ('.png', '.jpg', '.jpeg', '.bmp', '.tiff') ):
        logger.warning("UNSUPPORTED_IMAGE_FORMAT - File: %s", img_path)
    
    logger.info("PATHS_VALIDATED - Paths are valid")
    return True