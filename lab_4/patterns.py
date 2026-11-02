"""Модуль с регулярным выражением для парсинга логов."""

import re

LOG_PATTERN = re.compile(
    r"""
    ^                                                      
    (?P<ip>\d{1,3}(?:\.\d{1,3}){3})                      
    \s+(?P<identd>\S+)\s+(?P<userid>\S+)                  
    \s+\[(?P<date>                                         
        \d{2}/[A-Z][a-z]{2}/\d{4}                      
        :\d{2}:\d{2}:\d{2}                            
        \s[+-]\d{4}                                      
    )\]                                                   
    \s+"(?P<method>[A-Z]+)\s+(?P<url>[^\s"]+)\s+HTTP/\d\.\d" 
    \s+(?P<status>\d{3})\s+(?P<size>\d+)                   
    """,
    re.VERBOSE
)