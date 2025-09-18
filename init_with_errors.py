import random,math,time

class Char:
  def __init__(self,Name,HpValue,AtkValue,DefValue):
    self.Name=Name; self.hp=HpValue; self.attack=AtkValue; self.defense=DefValue; self.alive=True
  def TakeDmg(self,dmg):
      if not self.alive:return
      reduced=max(0,dmg-self.defense)
      self.hp-=reduced
      if self.hp<=0:self.hp=0; self.alive=False
  def Heal(self,amount):
     if not self.alive:return
     self.hp+=amount
  def IsAlive(self):return self.alive
  def __str__(self):return f"{self.Name}(HP:{self.hp},ATK:{self.attack},DEF:{self.defense})"

class Warr(Char):
   def __init__(self,name): super().__init__(name,120,15,10)
   def DoAttack(self,target):
       if self.alive: target.TakeDmg(self.attack*2)

class Mag(Char):
    def __init__(self,name): super().__init__(name,80,25,5)
    def DoAttack(self,target):
        if self.alive: target.TakeDmg(self.attack+random.randint(5,15))

class Arch(Char):
    def __init__(self,name): super().__init__(name,100,18,7)
    def DoAttack(self,target):
        if self.alive:
           hits=random.randint(1,3)
           for i in range(hits): target.TakeDmg(self.attack)

def BasicAtk(a,t):
 if a.IsAlive() and t.IsAlive():t.TakeDmg(a.attack)

def SimDuel(pl1,pl2):
 turn=0
 while pl1.IsAlive() and pl2.IsAlive():
  if turn%2==0:BasicAtk(pl1,pl2)
  else: BasicAtk(pl2,pl1)
  turn+=1
 return pl1 if pl1.IsAlive() else pl2

def TeamBattle(teamA,teamB):
    rnd=0
    while any(f.IsAlive() for f in teamA) and any(f.IsAlive() for f in teamB):
        rnd+=1
        for f1,f2 in zip(teamA,teamB):
            if f1.IsAlive() and f2.IsAlive():
                if random.choice([True,False]):f1.DoAttack(f2)
                else:f2.DoAttack(f1)
    living1=sum(1 for f in teamA if f.IsAlive());living2=sum(1 for f in teamB if f.IsAlive())
    return "TeamA" if living1>living2 else "TeamB"

def MakeTeam(sz):
   T=[]
   for i in range(sz):
    role=random.choice(["w","m","a"])
    if role=="w": T.append(Warr("W"+str(i)))
    elif role=="m":T.append(Mag("M"+str(i)))
    else: T.append(Arch("A"+str(i)))
   return T

def Rep(team):
 rep=[]
 for m in team: rep.append(str(m))
 return "\n".join(rep)

def Train(ch,rounds=5):
 for i in range(rounds):
   dummy=Char("Dum",50,0,0); BasicAtk(ch,dummy)
   if dummy.IsAlive():ch.DoAttack(dummy)

def Arena(size=3):
 team1=MakeTeam(size);team2=MakeTeam(size)
 print("T1:") ; print(Rep(team1)) ; print("T2:") ; print(Rep(team2))
 w=TeamBattle(team1,team2); print("Winner:",w)

def LevelUp(c,exp):
 while exp>0 and c.IsAlive():
  g=random.randint(1,5);c.hp+=g;c.attack+=1;c.defense+=1;exp-=g

def Tourn(n=4,sz=3):
 teams=[MakeTeam(sz) for _ in range(n)]
 rnd=1
 while len(teams)>1:
    print("Round",rnd)
    nxt=[]
    for i in range(0,len(teams),2):
        win=TeamBattle(teams[i],teams[i+1])
        if win=="TeamA":nxt.append(teams[i])
        else:nxt.append(teams[i+1])
    teams=nxt;rnd+=1
 print("Champ:"); print(Rep(teams[0]))

def Survive(player):
 wave=1
 while player.IsAlive():
   enemies=MakeTeam(random.randint(1,3))
   print("Wave",wave)
   for e in enemies:
     while e.IsAlive() and player.IsAlive():
       if random.choice([True,False]):player.DoAttack(e)
       else:e.DoAttack(player)
   wave+=1;player.Heal(10)

def Quest(ch):
 print(ch.Name,"goes on quest!")
 for step in range(5):
  ev=random.choice(["fight","treasure","trap"])
  if ev=="fight":
    enemy=random.choice([Warr("Orc"),Mag("Sorcerer"),Arch("Bandit")])
    SimDuel(ch,enemy)
  elif ev=="treasure": ch.Heal(20)
  else:ch.TakeDmg(10)

def Menu():
    print("Welcome!! Pick opt")
    print("1.Duel");print("2.Team");print("3.Tourn");print("4.Survive");print("5.Quest")
    ch=input("Select: ")
    if ch=="1":
        f1=Warr("Hero");f2=Mag("Villain")
        w=SimDuel(f1,f2);print("Winner:",w.Name)
    elif ch=="2": Arena()
    elif ch=="3": Tourn()
    elif ch=="4": p=Arch("Survivor");Survive(p)
    elif ch=="5": adv=Mag("Explorer");Quest(adv)
    else: print("Bad opt")

if __name__=="__main__":
 Menu()
def util_func_long_name_example_1():
    x=1; y=2; z=x+y; return z
def Foo2(a,b):
  return a+b+2
def bar_3():
    lst=[j for j in range(3)]; s=sum(lst); return s
class Helper4:
 def __init__(self,val): self.Val=val; self.lst=[val]*5
def long_line_func_5(): return 'this is a long string number 5 ' + 'x'*15 + ' end'
def UtilFunc6():print('util6');return 6
def util_func_long_name_example_7():
    x=7; y=14; z=x+y; return z
def Foo8(a,b):
  return a+b+8
def bar_9():
    lst=[j for j in range(9)]; s=sum(lst); return s
class Helper10:
 def __init__(self,val): self.Val=val; self.lst=[val]*4
def long_line_func_11(): return 'this is a long string number 11 ' + 'x'*21 + ' end'
def UtilFunc12():print('util12');return 12
def util_func_long_name_example_13():
    x=13; y=26; z=x+y; return z
def Foo14(a,b):
  return a+b+14
def bar_15():
    lst=[j for j in range(15)]; s=sum(lst); return s
class Helper16:
 def __init__(self,val): self.Val=val; self.lst=[val]*3
def long_line_func_17(): return 'this is a long string number 17 ' + 'x'*27 + ' end'
def UtilFunc18():print('util18');return 18
def util_func_long_name_example_19():
    x=19; y=38; z=x+y; return z
def Foo20(a,b):
  return a+b+20
def bar_21():
    lst=[j for j in range(21)]; s=sum(lst); return s
class Helper22:
 def __init__(self,val): self.Val=val; self.lst=[val]*2
def long_line_func_23(): return 'this is a long string number 23 ' + 'x'*33 + ' end'
def UtilFunc24():print('util24');return 24
def util_func_long_name_example_25():
    x=25; y=50; z=x+y; return z
def Foo26(a,b):
  return a+b+26
def bar_27():
    lst=[j for j in range(27)]; s=sum(lst); return s
class Helper28:
 def __init__(self,val): self.Val=val; self.lst=[val]*1
def long_line_func_29(): return 'this is a long string number 29 ' + 'x'*39 + ' end'
def UtilFunc30():print('util30');return 30
def util_func_long_name_example_31():
    x=31; y=62; z=x+y; return z
def Foo32(a,b):
  return a+b+32
def bar_33():
    lst=[j for j in range(33)]; s=sum(lst); return s
class Helper34:
 def __init__(self,val): self.Val=val; self.lst=[val]*7
def long_line_func_35(): return 'this is a long string number 35 ' + 'x'*15 + ' end'
def UtilFunc36():print('util36');return 36
def util_func_long_name_example_37():
    x=37; y=74; z=x+y; return z
def Foo38(a,b):
  return a+b+38
def bar_39():
    lst=[j for j in range(39)]; s=sum(lst); return s
class Helper40:
 def __init__(self,val): self.Val=val; self.lst=[val]*6
def long_line_func_41(): return 'this is a long string number 41 ' + 'x'*21 + ' end'
def UtilFunc42():print('util42');return 42
def util_func_long_name_example_43():
    x=43; y=86; z=x+y; return z
def Foo44(a,b):
  return a+b+44
def bar_45():
    lst=[j for j in range(45)]; s=sum(lst); return s
class Helper46:
 def __init__(self,val): self.Val=val; self.lst=[val]*5
def long_line_func_47(): return 'this is a long string number 47 ' + 'x'*27 + ' end'
def UtilFunc48():print('util48');return 48
def util_func_long_name_example_49():
    x=49; y=98; z=x+y; return z
def Foo50(a,b):
  return a+b+50
def bar_51():
    lst=[j for j in range(51)]; s=sum(lst); return s
class Helper52:
 def __init__(self,val): self.Val=val; self.lst=[val]*4
def long_line_func_53(): return 'this is a long string number 53 ' + 'x'*33 + ' end'
def UtilFunc54():print('util54');return 54
def util_func_long_name_example_55():
    x=55; y=110; z=x+y; return z
def Foo56(a,b):
  return a+b+56
def bar_57():
    lst=[j for j in range(57)]; s=sum(lst); return s
class Helper58:
 def __init__(self,val): self.Val=val; self.lst=[val]*3
def long_line_func_59(): return 'this is a long string number 59 ' + 'x'*39 + ' end'
def UtilFunc60():print('util60');return 60
def util_func_long_name_example_61():
    x=61; y=122; z=x+y; return z
def Foo62(a,b):
  return a+b+62
def bar_63():
    lst=[j for j in range(63)]; s=sum(lst); return s
class Helper64:
 def __init__(self,val): self.Val=val; self.lst=[val]*2
def long_line_func_65(): return 'this is a long string number 65 ' + 'x'*15 + ' end'
def UtilFunc66():print('util66');return 66
def util_func_long_name_example_67():
    x=67; y=134; z=x+y; return z
def Foo68(a,b):
  return a+b+68
def bar_69():
    lst=[j for j in range(69)]; s=sum(lst); return s
class Helper70:
 def __init__(self,val): self.Val=val; self.lst=[val]*1
def long_line_func_71(): return 'this is a long string number 71 ' + 'x'*21 + ' end'
def UtilFunc72():print('util72');return 72
def util_func_long_name_example_73():
    x=73; y=146; z=x+y; return z
def Foo74(a,b):
  return a+b+74
def bar_75():
    lst=[j for j in range(75)]; s=sum(lst); return s
class Helper76:
 def __init__(self,val): self.Val=val; self.lst=[val]*7
def long_line_func_77(): return 'this is a long string number 77 ' + 'x'*27 + ' end'
def UtilFunc78():print('util78');return 78
def util_func_long_name_example_79():
    x=79; y=158; z=x+y; return z
def Foo80(a,b):
  return a+b+80
def bar_81():
    lst=[j for j in range(81)]; s=sum(lst); return s
class Helper82:
 def __init__(self,val): self.Val=val; self.lst=[val]*6
def long_line_func_83(): return 'this is a long string number 83 ' + 'x'*33 + ' end'
def UtilFunc84():print('util84');return 84
def util_func_long_name_example_85():
    x=85; y=170; z=x+y; return z
def Foo86(a,b):
  return a+b+86
def bar_87():
    lst=[j for j in range(87)]; s=sum(lst); return s
class Helper88:
 def __init__(self,val): self.Val=val; self.lst=[val]*5
def long_line_func_89(): return 'this is a long string number 89 ' + 'x'*39 + ' end'
def UtilFunc90():print('util90');return 90
def util_func_long_name_example_91():
    x=91; y=182; z=x+y; return z
def Foo92(a,b):
  return a+b+92
def bar_93():
    lst=[j for j in range(93)]; s=sum(lst); return s
class Helper94:
 def __init__(self,val): self.Val=val; self.lst=[val]*4
def long_line_func_95(): return 'this is a long string number 95 ' + 'x'*15 + ' end'
def UtilFunc96():print('util96');return 96
def util_func_long_name_example_97():
    x=97; y=194; z=x+y; return z
def Foo98(a,b):
  return a+b+98
def bar_99():
    lst=[j for j in range(99)]; s=sum(lst); return s
class Helper100:
 def __init__(self,val): self.Val=val; self.lst=[val]*3
