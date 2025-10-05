from pandas import DataFrame
import re

def validation_check(pattern:str,string:str)->bool:
    """
    The function checks the string for validity
    :param pattern: regex pattern
    :param string: string to check
    :return: True if string is INVALID, False if valid
    """
    return not bool(re.fullmatch(pattern, str(string)))


def pattern_fun(patterns:dict,data_frame: DataFrame)->list[int]:
    """
    Finds rows with invalid values according to patterns
    :param patterns: dictionary {column_name: regex_pattern}
    :param data_frame: dataframe to check
    :return: list of row indexes with invalid data
    """
    row_numbers=[]
    for col_name, pattern in patterns.items():

        for index in data_frame.index:
            if validation_check(pattern,data_frame.loc[index,col_name]):

                row_numbers.append(index)


    return row_numbers


