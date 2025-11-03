import argparse

def get_parse() -> tuple:
    """
    Parsing arguments
    :return: tuple of pathes
    """
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('img_path', type = str, help = 'Path to image')
        parser.add_argument('save_path', type = str, help = 'Path to file')
        args = parser.parse_args()
        return args.img_path, args.save_path
    except:
        raise SyntaxError("Invalid data")
    
    