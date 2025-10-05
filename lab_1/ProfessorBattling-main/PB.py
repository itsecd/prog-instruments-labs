import os
import random

class battler:
    btNum = 0
    name = ''
    #STATS
    #Hp - Health Points / At - Attack / Df - Defense / Sp - Speed
    hp = 0
    inithp = 0
    at = 0
    df = 0
    sp = 0

    #TYPE
    #Pf - Professor Type / St - Student Type / DC - Department Chair Type / WD - Web Dev Type
    tp = ['','']

    #Moves
    mvs = []
    
    #User controlled. 1 if true, 0 if false
    userControl = 0

    def __init__(self, btNum, name, hp, at, df, sp, tp, mvs):
        self.name = name
        self.hp = int(hp)
        self.inithp = int(hp)
        self.at = int(at)
        self.df = int(df)
        self.sp = int(sp)
        self.btNum = int(btNum)
        self.tp = tp
        self.mvs = mvs

    def battlerDesc(self):
        tpDisplay = ''
        for battleType in self.tp:
            tpDisplay += " " + battleType
        tpDisplay = tpDisplay.upper()

        if (len(self.name) <= 3):
            print(f"{self.btNum}. {self.name}\t\tHP:{self.hp}  AT:{self.at}  DF:{self.df}  SP:{self.sp}  TYPE:{tpDisplay}")
        else:
            print(f"{self.btNum}. {self.name}\tHP:{self.hp}  AT:{self.at}  DF:{self.df}  SP:{self.sp}  TYPE:{tpDisplay}")

    def battlerStatus(self):
        displayNumHP = f"{self.hp:.2f}"
        displayHP = ""
        if self.hp <= 0:
            displayHP = "[           ]"
            displayNumHP = "KNOCKED OUT"
        else:
            displayHP = "["
            percentOfHealth = (self.hp / self.inithp) * 10
            # 80% Health = percentOfHealth = 8.0
            numberOfBars = percentOfHealth

            if percentOfHealth <= 10:
                while numberOfBars > 0:
                    displayHP += "/"
                    numberOfBars -= 1

                numberOfBars = int(10 - percentOfHealth)

                while numberOfBars > 0:
                    displayHP += " "
                    numberOfBars -= 1
            else:
                displayHP += "//////////"

            displayHP += "]"

        print(displayHP)
        print(f"HP:{displayNumHP}  AT:{self.at:.2f}  DF:{self.df:.2f}  SP:{self.sp:.2f}")
class move:
    number = 0      #Identifer of Move
    name = ''
    dmg = 0         #dmg is how much the move does damage wise
    acc = 0         #acc = How likely it is to hit
    eat = ''        #eat = Effective Against Type
    nat = ''        #nat = Not Effective Type
    sto = []        #sto = Status inflicted with this move on Oponent ['stat', numberAffectingStat]
    sts = []        #sts = Status inflicted with this move on Self       
    des = ''

    def __init__(self, number, name, dmg, acc, eat,nat, sto, sts, des):
        self.number = int(number)
        self.name = name
        self.dmg = int(dmg)
        self.acc = int(acc)
        self.eat = eat
        self.nat = nat
        self.sto = sto
        self.sts = sts

        eatDisplay = ''
        natDisplay = ''
        stoDisplay = ''
        stsDisplay = ''

        if self.eat == '?':
            eatDisplay = 'None'
        else:
            eatDisplay = self.eat

        if self.nat == '?':
            natDisplay = 'None'
        else:
            natDisplay = self.nat

        if self.sto == ['?','?']:
            stoDisplay = "None"
        else:
            stoDisplay = str(self.sto[0]) + " " + str(self.sto[1])

        if self.sts == ['?','?']:
            stsDisplay = "None"
        else:
            stsDisplay = str(self.sts[0]) + " " + str(self.sts[1])
        
        self.des = des + (f"\n\nDamage:{self.dmg} / Accuracy:{self.acc} \nEffective Against: {eatDisplay} / Not Effective Against: {natDisplay} \nStatus To Self: {stsDisplay} / Status to Opponent: {stoDisplay}")
def intro(battlers):
    accAnswers = ['1','2','3','4','5','6']  # List of acceptible answers to choose a battler
    userInput = ''                          # Allocates the input to one variable across the whole program
    answered = 0                            # Loop controller. 0 Until acceptible answer is given

    print("Welcome to the BATTLE ZONE")

    print("To begin, please select the number of who you'd like to battle with today")
    print("At any point if you see a >, press ENTER to continue")
    print("To learn about the stats, enter 0\n")

    
    #While a good answer hasn't been given yet
    while (answered == 0):
        #print the stat option
        print("0. Stats Info")

        #Print out each battler
        for battler in battlers:
            battler.battlerDesc()

        print("\n")

        #prompt for battler selection
        userInput = input("What would you like to select? : ")

        #If an approrpriate Answer was given
        if (userInput in accAnswers):
            answered = 1
        #If the user asked for stats info
        elif (userInput == "0"):
            #Clear screen, print stats info
            os.system('cls')
            print("\nThere are a total of 4 stats you should be concerned with in the BATTLE ZONE")
            print("HP - This is health points. When this hits 0, you're dead. Some battlers are proficent in this stat. Range: 0 - 500")
            print("AT - This is your attacking power. Higher Attacking power, the more damage you output. After comparing against the defenders DF, it removes their health. Range: 0 - 100")
            print("DF - This is your defensive power. Higher Defense can make your opponent's attacks weaker against you, so you lose less health. Range: 0 - 100")
            print("SP - This is your speed. This determines who goes first in a round. The faster you are, the faster you can get your hit in. Range: 0 - 50\n")
        #If invalid input is given
        else:
            os.system('cls')
            print("\nInvalid Input. Please enter the number associated with the battler you'd like to select\n")

    return (int(userInput) - 1)
def determineOpponent(battlers,userBattler):
    #Loop until a good opponent has been chosen
    answered = 0

    #Allocate for the choice
    randomChoice = 0

    #Lop until a good oponent
    while answered == 0:
        #Generate random number
        randomChoice = random.randint(0,5)

        #If ther andom choice doesn't match the user's
        if randomChoice != userBattler.btNum - 1:
            #Break loop, we found our oponent
            answered = 1

    #Display oponent and return the oponent
    print(f"Opponent chosen{randomChoice}")
    return randomChoice
def readBattlersFile():
    battlers = []
    with open('battlers.txt') as bFile:

        allBattlersLine = bFile.readlines()

        battlerCounter = 0
        while battlerCounter < len(allBattlersLine) - 1:
            battlerCounter += 1
            battlerLine = allBattlersLine[battlerCounter].replace(" ", "").split("/")

            battlers.append(battler(battlerLine[0], battlerLine[1], battlerLine[2], battlerLine[3], battlerLine[4], battlerLine[5], battlerLine[6].split("-"), battlerLine[7].split("-")))


    return battlers
def readMovesFile():
    moves = []
    with open('moves.txt') as mFile:
        allMovesLine = mFile.readlines()

        moveCounter = 0
        while moveCounter < len(allMovesLine) - 1:
            moveCounter += 1
            moveLine = allMovesLine[moveCounter].replace(" ", "").split("/")
            moves.append(move(moveLine[0], moveLine[1].replace("_"," "), moveLine[2], moveLine[3], moveLine[4], moveLine[5],moveLine[6].split("|"), moveLine[7].split("|"), moveLine[8].replace("_"," ")))

    return moves
def determineBattlers():

    battlers = readBattlersFile()
    moves = readMovesFile()

    i = 0
    while i < len(battlers) - 1:
        j = 0
        #battlers[i].battlerDesc()
        while j < len(battlers[i].mvs):
           battlers[i].mvs[j] = moves[int(battlers[i].mvs[j])]
           j += 1
        i += 1

    #Determine the user's battler by calling the intro (Which walks them through selecting one)
    userBtl = battlers[intro(battlers)]
    #Declare the user's battler as under their control
    userBtl.userControl = 1
    #Return the user's battler, and the oponent by calling referincing the battlers list with the index returned from determineOponent
    return([userBtl,battlers[determineOpponent(battlers, userBtl)]])
def dealAttack(target, origin, move):
    # Determine Accuracy. If the number is greater than the accuracy number, it misses. If it is the same or smaller, it hits
    rnd = random.randint(1, 100)

    # If who you're trying to hit is faster than you
    if target.sp > origin.sp:
        accuracyModifer = target.sp / 5.5
        rnd = rnd + accuracyModifer

    #Store the original move damage
    tempMoveDmg = move.dmg

    #If the move hits!
    if rnd <= move.acc:
        #If its hitting a target its supereffective on, boost the damange
        if move.eat in target.tp:
            move.dmg *= 2.5
        #If its hitting a target its not very effective on, weaken the damage
        elif move.nat in target.tp:
            move.dmg /= 2.5

        #If the attack of the origin is 0, set it to 0.000000000001 so there isn't a divide by 0 error
        if origin.at == 0:
            origin.at = 0.1
        if target.df == 0:
            target.df = 0.1
        if move.dmg == 0:
            move.dmg = 0.00000001

        # Calculate Damage USER ATTACK * (MOVE DAMAGE / DEFENSE OF TARGET)
        dmg = (origin.at / 2) * (move.dmg / (target.df))

        # Remove the damage from health
        target.hp -= dmg

        # Explain what happened to the user
        print(f"{origin.name}'s {move.name} did {dmg:.2f} damage!")

        # If the Status Against Oponent ISNT empty
        if move.sto[0] != ' ':
            #If the Status Against Oponent is ATTACK
            if move.sto[0] == 'at':
                print(f"{move.name} did {move.sto[1]} to {target.name}'s Attack")

                if target.at <= 0:
                    print("\nAttack Already As Low As It Can Go!")
                else:
                    target.at += int(move.sto[1])

                if target.at < 0:
                    target.at = 0

            #If the Status Against Oponent is DEFENSE
            elif move.sto[0] == 'df':

                #Explain to user what the effect is
                print(f"{move.name} did {move.sto[1]} to {target.name}'s Defense")

                #If the defense is already 0 or lower, explain it to user
                if target.df <= 0:
                    print("\nDefense Already As Low As It Can Go!")
                # If the defense isn't 0 or lower, lower the defense
                else:
                    target.df += int(move.sto[1])

                #After lowering it, if the defense is less than 0, set it TO 0
                if target.df < 0:
                    target.df = 0

            #If the Status Against Oponent is SPEED
            elif move.sto[0] == 'sp':
                #Explain Effect To User
                print(f"{move.name} did {move.sto[1]} to {target.name}'s Speed")

                #If Defense is already <0 don't decrease
                if target.df <= 0:
                    print("\nSpeed Already As Low As It Can Go!")
                # Else decrease it
                else:
                    target.sp += int(move.sto[1])

                #Check to see if its less than 0 after decreasing it
                if target.sp < 0:
                    target.sp = 0

            #If the Status Against Oponent is HEALTH
            elif move.sto[0] == 'hp':
                #Explain effect
                print(f"{move.name} did {move.sto[1]} to {target.name}'s HP")

                #Lower Health
                target.hp += int(move.sto[1])

        #If the move Status Against Self isn't Empty
        if move.sts[0] != ' ':
            #If the Status Against Self is ATTACK
            if move.sts[0] == 'at':
                #Explain effect
                print(f"{move.name} did {move.sts[1]} to {origin.name}'s Attack")

                #If Stat is over CAP
                if origin.at >= 100:
                    print("\nAttack Already As High As It Can Go!")
                #If stat ISNT over CAP
                else:
                    origin.at += int(move.sts[1])

                #Check to see if CAP has been breached after adding stats
                if target.at > 100:
                    target.at = 100
             #If the Status Against Self is DEFENSE
            elif move.sts[0] == 'df':
                #Explain effect
                print(f"{move.name} did {move.sts[1]} to {origin.name}'s Defense")

                #If Stat is over CAP
                if origin.df >= 100:
                    print("\nDefense Already As High As It Can Go!")
                #If stat ISNT over CAP
                else:
                    origin.df += int(move.sts[1])

                #Check to see if CAP has been breached after adding stats
                if target.df > 100:
                    target.df = 100
            #If the Status Against Self is SPEED
            elif move.sts[0] == 'sp':
                #Explain effect
                print(f"{move.name} did {move.sts[1]} to {origin.name}'s Speed")

                #If Stat is over CAP
                if origin.sp >= 50:
                    print("\nSpeed Already As High As It Can Go!")
                #If stat ISNT over CAP
                else:
                    origin.sp += int(move.sts[1])

                #Check to see if CAP has been breached after adding stats
                if target.sp > 50:
                    target.sp = 50
            #If the Status Against Self is HEALTH
            elif move.sts[0] == 'hp':
                #Explain effect
                print(f"{move.name} did {move.sts[1]} to {origin.name}'s Hp")
                #Add Stat
                origin.hp += int(move.sts[1])
    else:
        print(f"{move.name} missed!")

    move.dmg = tempMoveDmg
def askAttack(battlers, roundCount):
    acceptibleAnswers = ['1','2','3','4']       #All Answers that are acceptible
    answered = 0                                #Loop until this is 1
    chosenMove = ''                             #Allocate for chosen Move
    
    #Loop until good answer is given
    while answered == 0:
        #Prompt for vhoice
        print("\nPlease enter the number associated with your choice\n")
        print("0 - Learn About Your Moves")

        #Loop for every move and print out their info
        for num, move in enumerate(battlers[0].mvs):
            print(f"{num + 1} - {move.name}")
        
        #Gain the user's input
        userChoice = input("\nYour Choice: ")

        #If the user's choice is acceptible, select the chosen Move
        if userChoice in acceptibleAnswers:
            answered = 1
            chosenMove = battlers[0].mvs[int(userChoice) - 1]
        #If the user wants to learn about the moves
        elif userChoice == '0':
            #Prompt for what individual move they want to know about

            userChoice = input("Please enter the move you'd like to know about: ")
            os.system('cls')
            displayHealthBars(roundCounter, battlers)
            #If the choice is an acceptible answer

            if userChoice in acceptibleAnswers:
                #Output the description of the move
                userChoice = int(userChoice) - 1
                print(f"\n{userChoice + 1} - {battlers[0].mvs[userChoice].name} - {battlers[0].mvs[userChoice].des}")
            
            input("\nPress ENTER To Continue")
            os.system('cls')
            displayHealthBars(roundCounter, battlers)
        #If the input wasn't valid
        else:
            print("Invalid Input, Please Select A Move")

    dealAttack(battlers[1], battlers[0], chosenMove)
def AIAttack(battlers):
    #Random number to decide what move the AI will pick
    rnd = random.randint(0,3)
    
    #If the battler in the 0 position is the user controlled one
    if battlers[0].userControl == 1:
        print(f"{battlers[1].name} used {battlers[1].mvs[rnd].name}!")
        #Deal the attack
        dealAttack(battlers[0],battlers[1],battlers[1].mvs[rnd])
    #if the battler in the 1 position is the user controlled one
    else:
        print(f"{battlers[0].name} used {battlers[0].mvs[rnd].name}!")
        #Deal the attack
        dealAttack(battlers[1],battlers[0],battlers[0].mvs[rnd])
def displayHealthBars(roundCounter, battlers):
    print(f"ROUND {roundCounter}")

    #If the battler in the 0 position has the user control
    if battlers[0].userControl == 1:
        #Print the 0 battler first
        print(f"You - {battlers[0].name}")
        battlers[0].battlerStatus()
        print(f"\nOpon - {battlers[1].name}")
        battlers[1].battlerStatus()
    #If the battler in the 0 psotion ISNT user controlled
    else:
        #Print the 1 battler first
        print(f"You - {battlers[1].name}")
        battlers[1].battlerStatus()
        print(f"\nOpon - {battlers[0].name}")
        battlers[0].battlerStatus()

    print("\n")
def determineOrder(battlers):
    #This holds the ordered battlers
    orderBattlerHolder = []

    #If their speed is equal
    if battlers[0].sp == battlers[1].sp:
        #Random number
        rnd = random.randint(1,2)
        #If its 1, grab the first battler
        if rnd == 1:
            orderBattlerHolder = [battlers[0], battlers[1]]
        #If its not, grab the secondBattler and stick them first
        else:
            orderBattlerHolder = [battlers[1],battlers[0]]

    #If the 0 battler is faster, assign them first
    elif battlers[0].sp > battlers[1].sp:
        orderBattlerHolder = [battlers[0],battlers[1]]

    #If the 1 battler is faster, assign them first
    else:
        orderBattlerHolder = [battlers[1],battlers[0]]

    #Returned the ORDERED battlers
    return orderBattlerHolder
def checkForWin(battlers):
    #Check to see if either of the battlers has hit 0 or less HP
    if battlers[0].hp <= 0:
        return 1
    elif battlers[1].hp <= 0:
        return 1
    else:
        return 0
def makeRound(roundCounter, battlers):
    #This holds all of the battlers
    #battlerHolder = battlers

    #Ordering the battlers by speed
    orderBattlerHolder = determineOrder(battlers)

    #Clear Screen
    os.system('cls')

    #If the first attacker is under user Control
    if orderBattlerHolder[0].userControl == 1:
        #Before you play, check that no one has one yet
        if checkForWin(battlers) != 1:
            #Display Health Bars
            displayHealthBars(roundCounter, battlers)
            #Ask the User To Attack
            askAttack(battlers, roundCounter)
            #Pause
            input(">")
            #Check no one has one yet
            if checkForWin(battlers) != 1:
                #Clear screen
                os.system('cls')
                #Display health bars
                displayHealthBars(roundCounter, battlers)
                #Let the AI Attack
                AIAttack(battlers)
                #Check for win
                checkForWin(battlers)
                #Pause
                input(">")
    #If the first attack isn't under user control
    else:  
        #Check for a win
        if checkForWin(battlers) != 1:
            #Display health bars
            displayHealthBars(roundCounter, battlers)
            #Let the AI go
            AIAttack(battlers)
            #pause
            input(">")
            #Check for Win
            if checkForWin(battlers) != 1:
                #Clear Screen
                os.system('cls')
                #Display Health Bars
                displayHealthBars(roundCounter, battlers)
                #Ask User to Attack
                askAttack(battlers, roundCounter)
                #Pause
                input(">")
def displayWinner(battlers):
    #If the winner is battler1
    if battlers[0].hp <= 0:
        #If the winning battler is controlled by the user
        if battlers[1].userControl == 1:
            print(f"Congrats! You and {battlers[1].name} Won!")
        #If the winning battler was the AI
        else:
            print(f"The Winner Is {battlers[1].name}!")
    #If the winner is battler0
    else:
        #If the winning battler is controlled by the user
        if battlers[0].userControl == 1:
            print(f"Congrats! You and {battlers[0].name} Won!")
        #If the winning battler was AI
        else:
            print(f"The Winner Is {battlers[0].name}!")

#Lets decide what stats are GOOD

# Attack and Defense on a scale of 0-100
# Health is on a scale of 1-500
# Speed is on a scale of 0-50

# Dr.Ford   - + Attack  + Speed 
# Jackson   - + Speed   - Defense
# Markley   - + Health  - Attack
# Ben       - + Attack  - Speed
# Student-A - + Defense - Health
# Student-B - + Attack  - Defense

#loops until the game has finished
gameActive = 1
#This counts the rounds
roundCounter = 1

#This calls the determine Battlers, which calls intro and helps the user pick their battler 
# and calls determine opponent to pick the other battler
battlers = determineBattlers()

#Clear the screen
os.system('cls')

#Print out the battlers
print(f"Your battler  is {battlers[0].name}")
print(f"Your opponent is {battlers[1].name}")

#Pause until the user continues
input(">")

#loop for as long as the game is active
while gameActive == 1:
    #Make a round
    makeRound(roundCounter, battlers)
    #Add a count to the round
    roundCounter+=1

    #If someone's HP hit 0
    if checkForWin(battlers) == 1:
        os.system('cls')                    #Clear screen
        displayHealthBars("END", battlers)  #Display a final health / stat bar
        displayWinner(battlers)             #Display winners
        gameActive = 0                      #Break Loop

input("Thank you for playing the BATTLE ZONE!")