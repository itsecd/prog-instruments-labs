#Menu of Functions
import sys
balance_a=0
gas_a= 30
water_a = 100
electric_a = 270
def init():
  global balance_a
  balance_a=250.00
print ("------Welcome to the Bank ATM Program------")
print ()
print ("---Login---")
tries= 3
pin_code= "1234"
pin_try= input ("Please enter you pin to access your bank account > ")
while pin_code!=pin_try:
  tries=tries-1
  if (tries==2):
    print ("Wrong Pin! You have 2 attempts remaining.")
    pin_try= input ("Please, enter your pin: ")
  elif (tries==1):
    print ("Wrong Pin! You have 1 attempts remaining. Be Careful!")
    pin_try= input ("Please, enter your pin: ")
  elif (tries==0):
   print ("Your attempts overed. Bank account locked! Please contact with your bank.")
   sys.exit
   SystemExit()
print ()
print ("Your Pin Code correct!")
print ("----------------------")
  
def Balance():# first function
  global balance_a
  print()
  print ("Your Balance : ",end="")
  print ( "%.2f" % balance_a)
  

  
def Deposit ():# second function
  print ()
  print ("---Deposit Menu---")
  global balance_a
  Balance()
  print ()
  print("How much dou you want to deposit to your bank account?"  ,end="")
  deposittoaccount = float(input (" > "))
  print ()
  print ("%.2f" %deposittoaccount , "deposited to your bank account.")
  balance_a=balance_a +deposittoaccount
  Balance()
  
def Withdraw():#third function
  print()
  print ("---Withdraw Menu---")
  global balance_a
  Balance()
  print ()
  decrease = float(input("How much do you want to withdraw from your account? > "))
  if balance_a-decrease > 0:
    print()
    print ("%.2f"%decrease+" has been deducted from your bank account.")
    balance_a= balance_a-decrease
  else:
    print()
    print ("Insufficient balance for this transaction!")
  Balance()




def PayBillOnline():#fourth function
  global balance_a
  global electric_a
  global gas_a
  global water_a
  continue_bill = "yes"
  print ()
  print ("---Pay Bill Online Menu---")
  Balance()
  while continue_bill == "yes":
    print () 
    print ("--Your Bills--")
    print("1. Electric Bill - "+"%.2f"%electric_a)
    print("2. Gas Bill - "+"%.2f"%gas_a)
    print("3. Water Bill - "+"%.2f"%water_a)
    print("4. Exit Pay Bills")
    print("Please select a number between 1 and 4 > ",end="")
    menu_option = input()
    if menu_option<"1" or menu_option>"4":
     print ()
     print("Input Error. Please enter a number between 1 and 4.")
    elif menu_option=="1":
      if (balance_a-electric_a>0):
        balance_a = balance_a-electric_a
        electric_a = 0 
        print () 
        print ("Your electric bill paid. Thank you!")
        Balance()
      else:
       print ()
       print ("Insufficient balance. Electric bill could not paid!") 
    elif menu_option=="2":
      if (balance_a-gas_a>0):
        balance_a = balance_a-gas_a
        gas_a = 0
        print () 
        print ("Your gas bill paid. Thank you!")
        Balance()
      else:
       print () 
       print ("Insufficient balance. Gas bill could not paid!")
    elif menu_option=="3":
      if (balance_a-water_a>0):
        balance_a = balance_a-water_a
        water_a = 0 
        print () 
        print ("Your water bill paid. Thank you!")
        Balance()
      else:
       print () 
       print ("Insufficient balance. Water bill could not paid!")
    elif menu_option =="4":
      print ()
      print ("Your are returning to Bank Menu.")
      continue_bill = "no"
    else:
     print () 
     print("Input Error. Please enter a number between 1 and 4.")







  
init() 
Balance()
def menu():#menu function
  print()
  print("--- Bank Menu  ---")
  print("(1) View Your Balance")
  print("(2) Deposit Cash")
  print("(3) Withdraw Cash")
  print("(4) Pay Bill Online")
  print ("(5) Quit")
  print ("Please choose (1-5) > ")
#while loop  
repeat="yes"
while repeat =="yes":
  menu()
  choice =input()
  if choice<"1" or choice>"5":
    print ()
    print("Input Error. Please enter a number between 1 and 5.")
  if choice =="1":
    Balance()
  if choice =="2":
    Deposit()
  if choice =="3":
    Withdraw()
  if choice =="4":
    PayBillOnline()
  if choice=="5":
    print("Thanks for using the Bank ATM program. Good Bye!")
    repeat="no"
    SystemExit()

