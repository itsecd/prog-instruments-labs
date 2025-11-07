import argparse

def arg_parse() -> str:
    """
    Parsing arguments 
    :return: Annotation file name
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('annot_file_name', type = str, help = 'Name of annotation file')
    args = parser.parse_args()
    return args.annot_file_name