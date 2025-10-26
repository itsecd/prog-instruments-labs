from dataclasses import dataclass

@dataclass
class Entry:
    """Node interface"""
    telephone: str
    height: str
    inn: str
    passport_number: str
    university: str
    age: str
    academic_degree: str
    worldview: str
    address: str