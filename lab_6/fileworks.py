import json

def read_json(filename: str) -> dict:
    """
    Read data from json file

    Args:
        filename: Path to the JSON file to read
    
    Returns:
        dict: Parsed JSON data as a dictionary
    """
    try:
        with open(filename) as json_file:
            data = json.load(json_file)
        return data
    except Exception as e:
        print(e)
        raise e
    

def read_str(filename: str) -> str:
    """
    Read text file mode

    Args:
        filename: Path to the text file to read
    
    Returns:
        str: Content of the file as a string
    """
    try:
        with open(filename, encoding="UTF-8") as file:
            return file.read()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise e


def read_bytes(filename: str) -> bytes:
    """
    Read file in binary mode

    Args:
        filename: Path to the binary file to read
    
    Returns:
        bytes: Content of the file as bytes
    """
    try:
        with open(filename, 'rb') as file:
            return file.read()
    except FileNotFoundError as e:
        raise e
    

def str_writer(text: str, filename: str):
    """
    Write text to file
    
    Args:
        text: String content to write to the file
        filename: Path to the output file
    """
    try:
        with open(filename, 'w', encoding="UTF-8") as file:
            file.write(text)
    except Exception as e:
        print(f"Error: {e}")
        raise e


def encrypted_ChaCha20_writer(text: bytes, nonce: bytes, filename: str):
    """
    Write encrypted text and nonce
    
    Args:
        text: Encrypted text as bytes to write
        nonce: Nonce value to write before the encrypted text
        filename: Path to the output file
    """
    try:
        with open(filename, 'wb') as file:
            file.write(nonce)
            file.write(text)
    except Exception as e:
        print(f"Error: {e}")
        raise e