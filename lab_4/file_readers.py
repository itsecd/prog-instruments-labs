import json
import logging

logger = logging.getLogger(__name__)

def read_json_file(input_file: str):
    """This function reads file and returns json.load
    
    Parametres:
        input_file(str): path of the input file"""
    logger.info(f"Reading json file: {input_file}")
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.error(f"Failed to read json file: {exc}")
    

def write_file(output_file: str, output: list) -> None:
    """This function writes data into text file
    
    Parametres:
        output_file(str): path and name of the output file
        
        output(str): text that needed to be written"""
    logger.info(f"Writing data into file: {output_file}")
    try:
        with open(output_file, "w", encoding="utf-8") as o:
            for out in output:
                o.write(out)
                o.write("\n")
    except Exception as exc:
        logger.error(f"Failed to read txt file: {exc}")
