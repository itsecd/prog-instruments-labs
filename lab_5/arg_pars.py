import argparse
import datetime
import logging
import os
import sys
import time

from audit import audit_log


logger = logging.getLogger("image_processing.parser")


def get_parse() -> tuple:
    """
    Parsing arguments
    :return: tuple of pathes
    """
    program_start_time = time.time()
    logger.info("FUNCTION_STARTED - Parsing command line arguments")

    audit_log("PROGRAM_INITIALIZATION_STARTED", extra_data={
        "program_start_timestamp": datetime.datetime.now().isoformat(),
        "command_line_arguments": sys.argv,
        "python_version": sys.version,
        "executable_path": sys.executable
    })

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('img_path', type = str, help = 'Path to image')
        parser.add_argument('save_path', type = str, help = 'Path to file')
        args = parser.parse_args()

        logger.info("ARGUMENTS_PARSED - Image path: %s, Save path: %s", 
            args.img_path, 
            args.save_path
            )
        
        audit_log("ARGUMENT_PARSING_COMPLETED", extra_data={
            "parsed_arguments": {
                "input_image_path": args.img_path,
                "output_image_path": args.save_path,
                "arguments_count": 2,
                "input_file_extension": os.path.splitext(args.img_path)[1].lower(),
                "output_file_extension": os.path.splitext(args.save_path)[1].lower()
            },
            "parsing_duration_ms": round((time.time() - program_start_time) * 1000, 2)
        })
        
        return args.img_path, args.save_path
    except Exception as e:
        logger.error("ARGUMENT_PARSING_FAILED - Error: %s", str(e))
        audit_log("ARGUMENT_PARSING_FAILED", extra_data={
            "error_details": {
                "error_type": type(e).__name__,
                "error_message": str(e)
            },
            "arguments_received": sys.argv[1:] if len(sys.argv) > 1 else []
        })
        raise SyntaxError("Invalid data")
    

def validate_paths(img_path: str, save_path: str) -> bool:
    """
    Check paths are correct or not.
    :param img_path: path to image
    :param save_path: path to save
    :return: True if paths are valide
    """
    validation_start = time.time()
    logger.debug(
        "VALIDATING_PATHS - Input: %s, Output: %s", 
        img_path, 
        save_path
        )
    
    audit_log("PATH_VALIDATION_STARTED", extra_data={
        "validation_parameters": {
            "input_path": img_path,
            "output_path": save_path
        }
    })
    
    if not img_path or not save_path:
        logger.error("EMPTY_PATHS - One or both paths are empty")
        audit_log("PATH_VALIDATION_FAILED", extra_data={
            "input_path_empty": not bool(img_path),
            "output_path_empty": not bool(save_path)
        })
        raise ValueError("Paths cannot be empty")
    
    if not img_path.lower().endswith( ('.png', '.jpg', '.jpeg', '.bmp', '.tiff') ):
        logger.warning("UNSUPPORTED_IMAGE_FORMAT - File: %s", img_path)
    
    logger.info("PATHS_VALIDATED - Paths are valid")
    audit_log("PATH_VALIDATION_COMPLETED", extra_data={
        "validation_duration_ms": round((time.time() - validation_start) * 1000, 2),
        "validation_result": "SUCCESS"
    })
    return True