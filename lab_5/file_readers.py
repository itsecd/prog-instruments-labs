import json


def read_json_file(input_file: str):
    """This function reads file and returns json.load
    
    Parametres:
        input_file(str): path of the input file"""
    try:
        with open(input_file, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as exc:
        print(f"Failed to read json file: {exc}")


def write_key(output_file: str, key: bytes) -> None:
    """This function writes a key to a file
    
    Parametres:
        output_file(str): path of the output file
        
        key(bytes): key to write"""
    try:
        with open(output_file, "wb") as file:
            file.write(key)
    except Exception as exc:
        print(f"Failed to write to a file: {exc}")


def write_file(output_file: str, output: str) -> None:
    """This function writes data into text file
    
    Parametres:
        output_file(str): path and name of the output file
        
        output(str): text that needed to be written"""
    try:
        if type(output) is bytes:
            with open(output_file, "wb") as file:
                file.write(output)
        else:  
            with open(output_file, "w", encoding="utf-8") as file:
                file.write(output)
    except Exception as exc:
        print(f"Failed to read txt file: {exc}")


def read_txt(input_file: str):
    """This function reads txt data from file
    
    Parametres:
        input_file(str): path of the input file"""
    try:
        with open(input_file, "rb") as file:
            data = file.read()
        return data
    except Exception as exc:
        print(f"Failed reading file: {exc}")

