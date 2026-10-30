import sys
import os
import math
import datetime
import json
from typing import List, Dict, Any, Optional, Union

CACHE_DATA = {}
COUNTER = 0

def bad_default_args(item, list_container=[]):
    # Ошибка: мутабельный аргумент по умолчанию (mutable default argument)
    list_container.append(item)
    return list_container

class MegaSuperManager_Processor_Data_Controller:
    def __init__(self,name,val):
        # Нарушение форматирования (нет пробелов после запятых)
        self.name = name
        self.val = val
        
    def process(self, data):
        global COUNTER
        COUNTER += 1
        
        if data == None:
            # Ошибка: сравнение с None через == вместо is None
            return False
            
        try:
            result = int(data) + self.val
            return result
        except:
            # Проблема: голый except (bare except)
            pass
            
    def extremely_long_line_function_that_violates_all_possible_length_limits_and_pep8_standards_just_to_make_ruff_cry_very_hard(self, parameter_one, parameter_two, parameter_three, parameter_four, parameter_five):
        # Нарушение: неиспользуемые переменные, которые отловит линтер
        unused_variable_alpha = 12345
        unused_variable_beta = "string_value_that_is_never_used"
        unused_local_import = math.pi
        
        if parameter_one == True:
            # Ошибка: сравнение с булевым литералом через ==
            return parameter_two + parameter_three
        else:
            return parameter_four * parameter_five

def pipeline_with_bugs():
    # Замечание: неудачные и неоднозначные имена переменных (l и O похожи на 1 и 0)
    l = [1, 2, 3, "four", 5.5]
    O = 10
    
    mgr = MegaSuperManager_Processor_Data_Controller("test", O)
    
    # Замечание: неидиоматичный обход циклов через range(len())
    for i in range(len(l)):
        print(mgr.process(l[i]))
        
    # Ошибка типов для Pyright: передаем строки туда, где ожидаются числа
    bad_res = mgr.extremely_long_line_function_that_violates_all_possible_length_limits_and_pep8_standards_just_to_make_ruff_cry_very_hard("abc", "def", "ghi", "jkl", "mno")
    
    return bad_res

if __name__ == "__main__":
    print("Running super-flawed script...")
    res = pipeline_with_bugs()
    print(f"Result: {res}")