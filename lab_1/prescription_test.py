import unittest


from datetime import date, timedelta


from prescription import Prescription


class PrescriptionTest(unittest.TestCase):
    
    def test_completion_date(self):
        dispense_date = date.today() - timedelta(days = 15)
        prescription = Prescription( dispense_date, days_supply = 30 )
        self.assertEquals( date.today() + timedelta(days=15) , 
                          prescription.completion_date() )
        
    def test_days_supply(self):
        dispense_date = date.today() 
        prescription = Prescription( dispense_date , days_supply = 3 )
        self.assertEquals( 
            [ date.today() , date.today() + timedelta(days = 1) , 
             date.today()+timedelta(days =2 ) ] , prescription.days_taken() )
        
        
if __name__ == "__main__":
    unittest.main()