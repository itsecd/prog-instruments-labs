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

    def __init__(self, dic: dict):
        """Node constructor"""
        self.telephone = dic['telephone']
        self.height = dic['height']
        self.inn = dic['inn']
        self.passport_number = dic['passport_number']
        self.university = dic['university']
        self.age = dic['age']
        self.academic_degree = dic['academic_degree']
        self.worldview = dic['worldview']
        self.address = dic['address']