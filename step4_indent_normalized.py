import random, math, time

class Char:
    def __init__(self, name, hp_value, atk_value, def_value):
        self.name = name
        self.hp = hp_value
        self.attack = atk_value
        self.defense = def_value
        self.alive = True
    def take_dmg(self, dmg):
            if not self.alive:return
            reduced = max(0, dmg - self.defense)
            self.hp -= reduced
            if self.hp<=0:self.hp = 0
            self.alive = False
    def Heal(self, amount):
        if not self.alive:return
        self.hp += amount
    def is_alive(self):return self.alive
    def __str__(self):return f"{self.name}(HP:{self.hp}, ATK:{self.attack}, DEF:{self.defense})"

class Warr(Char):
    def __init__(self, name): super().__init__(name, 120, 15, 10)
    def do_attack(self, target):
            if self.alive: target.take_dmg(self.attack * 2)

class Mag(Char):
        def __init__(self, name): super().__init__(name, 80, 25, 5)
        def do_attack(self, target):
                if self.alive: target.take_dmg(self.attack + random.randint(5, 15))

class Arch(Char):
        def __init__(self, name): super().__init__(name, 100, 18, 7)
        def do_attack(self, target):
                if self.alive:
                    hits = random.randint(1, 3)
                    for i in range(hits): target.take_dmg(self.attack)

def basic_attack(a, t):
if a.is_alive() and t.is_alive():t.take_dmg(a.attack)

def simulate_duel(pl1, pl2):
turn = 0
while pl1.is_alive() and pl2.is_alive():
    if turn%2==0:basic_attack(pl1, pl2)
    else: basic_attack(pl2, pl1)
    turn += 1
return pl1 if pl1.is_alive() else pl2

def team_battle(teamA, teamB):
        rnd = 0
        while any(f.is_alive() for f in teamA) and any(f.is_alive() for f in teamB):
                rnd += 1
                for f1, f2 in zip(teamA, teamB):
                        if f1.is_alive() and f2.is_alive():
                                if random.choice([True, False]):f1.do_attack(f2)
                                else:f2.do_attack(f1)
        living1 = sum(1 for f in teamA if f.is_alive())
        living2 = sum(1 for f in teamB if f.is_alive())
        return "TeamA" if living1>living2 else "TeamB"

def make_team(sz):
    T = []
    for i in range(sz):
        role = random.choice(["w", "m", "a"])
        if role=="w": T.append(Warr("W" + str(i)))
        elif role=="m":T.append(Mag("M" + str(i)))
        else: T.append(Arch("A" + str(i)))
    return T

def rep(team):
rep = []
for m in team: rep.append(str(m))
return "\n".join(rep)

def train(ch, rounds = 5):
for i in range(rounds):
    dummy = Char("Dum", 50, 0, 0)
    basic_attack(ch, dummy)
    if dummy.is_alive():ch.do_attack(dummy)

def arena(size = 3):
team1 = make_team(size)
team2 = make_team(size)
print("T1:")
print(rep(team1))
print("T2:")
print(rep(team2))
w = team_battle(team1, team2)
print("Winner:", w)

def level_up(c, exp):
while exp>0 and c.is_alive():
    g = random.randint(1, 5)
    c.hp += g
    c.attack += 1
    c.defense += 1
    exp -= g

def tournament(n = 4, sz = 3):
teams = [make_team(sz) for _ in range(n)]
rnd = 1
while len(teams)>1:
        print("Round", rnd)
        nxt = []
        for i in range(0, len(teams), 2):
                win = team_battle(teams[i], teams[i + 1])
                if win=="TeamA":nxt.append(teams[i])
                else:nxt.append(teams[i + 1])
        teams = nxt
        rnd += 1
print("Champ:")
print(rep(teams[0]))

def survive(player):
wave = 1
while player.is_alive():
    enemies = make_team(random.randint(1, 3))
    print("Wave", wave)
    for e in enemies:
        while e.is_alive() and player.is_alive():
            if random.choice([True, False]):player.do_attack(e)
            else:e.do_attack(player)
    wave += 1
    player.Heal(10)

def quest(ch):
print(ch.name, "goes on quest!")
for step in range(5):
    ev = random.choice(["fight", "treasure", "trap"])
    if ev=="fight":
        enemy = random.choice([Warr("Orc"), Mag("Sorcerer"), Arch("Bandit")])
        simulate_duel(ch, enemy)
    elif ev=="treasure": ch.Heal(20)
    else:ch.take_dmg(10)

def menu():
        print("Welcome!! Pick opt")
        print("1.Duel")
        print("2.Team")
        print("3.tournament")
        print("4.survive")
        print("5.quest")
        ch = input("Select: ")
        if ch=="1":
                f1 = Warr("Hero")
                f2 = Mag("Villain")
                w = simulate_duel(f1, f2)
                print("Winner:", w.name)
        elif ch=="2": arena()
        elif ch=="3": tournament()
        elif ch=="4": p = Arch("Survivor")
        survive(p)
        elif ch == "5": adv = Mag("Explorer")
        quest(adv)
        else: print("Bad opt")

if __name__=="__main__":
menu()
def util_func_long_name_example_1():
        x = 1
        y = 2
        z = x + y
        return z
def foo_2(a, b):
    return a + b + 2
def bar_3():
        lst = [j for j in range(3)]
        s = sum(lst)
        return s
class Helper4:
def __init__(self, val): self.val = val
self.lst = [val]*5
def long_line_func_5(): return 'this is a long string number 5 ' + 'x'*15 + ' end'
def util_func_6():print('util6')
return 6
def util_func_long_name_example_7():
        x = 7
        y = 14
        z = x + y
        return z
def foo_8(a, b):
    return a + b + 8
def bar_9():
        lst = [j for j in range(9)]
        s = sum(lst)
        return s
class Helper10:
def __init__(self, val): self.val = val
self.lst = [val]*4
def long_line_func_11(): return 'this is a long string number 11 ' + 'x'*21 + ' end'
def util_func_12():print('util12')
return 12
def util_func_long_name_example_13():
        x = 13
        y = 26
        z = x + y
        return z
def foo_14(a, b):
    return a + b + 14
def bar_15():
        lst = [j for j in range(15)]
        s = sum(lst)
        return s
class Helper16:
def __init__(self, val): self.val = val
self.lst = [val]*3
def long_line_func_17(): return 'this is a long string number 17 ' + 'x'*27 + ' end'
def util_func_18():print('util18')
return 18
def util_func_long_name_example_19():
        x = 19
        y = 38
        z = x + y
        return z
def foo_20(a, b):
    return a + b + 20
def bar_21():
        lst = [j for j in range(21)]
        s = sum(lst)
        return s
class Helper22:
def __init__(self, val): self.val = val
self.lst = [val]*2
def long_line_func_23(): return 'this is a long string number 23 ' + 'x'*33 + ' end'
def util_func_24():print('util24')
return 24
def util_func_long_name_example_25():
        x = 25
        y = 50
        z = x + y
        return z
def foo_26(a, b):
    return a + b + 26
def bar_27():
        lst = [j for j in range(27)]
        s = sum(lst)
        return s
class Helper28:
def __init__(self, val): self.val = val
self.lst = [val]*1
def long_line_func_29(): return 'this is a long string number 29 ' + 'x'*39 + ' end'
def util_func_30():print('util30')
return 30
def util_func_long_name_example_31():
        x = 31
        y = 62
        z = x + y
        return z
def foo_32(a, b):
    return a + b + 32
def bar_33():
        lst = [j for j in range(33)]
        s = sum(lst)
        return s
class Helper34:
def __init__(self, val): self.val = val
self.lst = [val]*7
def long_line_func_35(): return 'this is a long string number 35 ' + 'x'*15 + ' end'
def util_func_36():print('util36')
return 36
def util_func_long_name_example_37():
        x = 37
        y = 74
        z = x + y
        return z
def foo_38(a, b):
    return a + b + 38
def bar_39():
        lst = [j for j in range(39)]
        s = sum(lst)
        return s
class Helper40:
def __init__(self, val): self.val = val
self.lst = [val]*6
def long_line_func_41(): return 'this is a long string number 41 ' + 'x'*21 + ' end'
def util_func_42():print('util42')
return 42
def util_func_long_name_example_43():
        x = 43
        y = 86
        z = x + y
        return z
def foo_44(a, b):
    return a + b + 44
def bar_45():
        lst = [j for j in range(45)]
        s = sum(lst)
        return s
class Helper46:
def __init__(self, val): self.val = val
self.lst = [val]*5
def long_line_func_47(): return 'this is a long string number 47 ' + 'x'*27 + ' end'
def util_func_48():print('util48')
return 48
def util_func_long_name_example_49():
        x = 49
        y = 98
        z = x + y
        return z
def foo_50(a, b):
    return a + b + 50
def bar_51():
        lst = [j for j in range(51)]
        s = sum(lst)
        return s
class Helper52:
def __init__(self, val): self.val = val
self.lst = [val]*4
def long_line_func_53(): return 'this is a long string number 53 ' + 'x'*33 + ' end'
def util_func_54():print('util54')
return 54
def util_func_long_name_example_55():
        x = 55
        y = 110
        z = x + y
        return z
def foo_56(a, b):
    return a + b + 56
def bar_57():
        lst = [j for j in range(57)]
        s = sum(lst)
        return s
class Helper58:
def __init__(self, val): self.val = val
self.lst = [val]*3
def long_line_func_59(): return 'this is a long string number 59 ' + 'x'*39 + ' end'
def util_func_60():print('util60')
return 60
def util_func_long_name_example_61():
        x = 61
        y = 122
        z = x + y
        return z
def foo_62(a, b):
    return a + b + 62
def bar_63():
        lst = [j for j in range(63)]
        s = sum(lst)
        return s
class Helper64:
def __init__(self, val): self.val = val
self.lst = [val]*2
def long_line_func_65(): return 'this is a long string number 65 ' + 'x'*15 + ' end'
def util_func_66():print('util66')
return 66
def util_func_long_name_example_67():
        x = 67
        y = 134
        z = x + y
        return z
def foo_68(a, b):
    return a + b + 68
def bar_69():
        lst = [j for j in range(69)]
        s = sum(lst)
        return s
class Helper70:
def __init__(self, val): self.val = val
self.lst = [val]*1
def long_line_func_71(): return 'this is a long string number 71 ' + 'x'*21 + ' end'
def util_func_72():print('util72')
return 72
def util_func_long_name_example_73():
        x = 73
        y = 146
        z = x + y
        return z
def foo_74(a, b):
    return a + b + 74
def bar_75():
        lst = [j for j in range(75)]
        s = sum(lst)
        return s
class Helper76:
def __init__(self, val): self.val = val
self.lst = [val]*7
def long_line_func_77(): return 'this is a long string number 77 ' + 'x'*27 + ' end'
def util_func_78():print('util78')
return 78
def util_func_long_name_example_79():
        x = 79
        y = 158
        z = x + y
        return z
def foo_80(a, b):
    return a + b + 80
def bar_81():
        lst = [j for j in range(81)]
        s = sum(lst)
        return s
class Helper82:
def __init__(self, val): self.val = val
self.lst = [val]*6
def long_line_func_83(): return 'this is a long string number 83 ' + 'x'*33 + ' end'
def util_func_84():print('util84')
return 84
def util_func_long_name_example_85():
        x = 85
        y = 170
        z = x + y
        return z
def foo_86(a, b):
    return a + b + 86
def bar_87():
        lst = [j for j in range(87)]
        s = sum(lst)
        return s
class Helper88:
def __init__(self, val): self.val = val
self.lst = [val]*5
def long_line_func_89(): return 'this is a long string number 89 ' + 'x'*39 + ' end'
def util_func_90():print('util90')
return 90
def util_func_long_name_example_91():
        x = 91
        y = 182
        z = x + y
        return z
def foo_92(a, b):
    return a + b + 92
def bar_93():
        lst = [j for j in range(93)]
        s = sum(lst)
        return s
class Helper94:
def __init__(self, val): self.val = val
self.lst = [val]*4
def long_line_func_95(): return 'this is a long string number 95 ' + 'x'*15 + ' end'
def util_func_96():print('util96')
return 96
def util_func_long_name_example_97():
        x = 97
        y = 194
        z = x + y
        return z
def foo_98(a, b):
    return a + b + 98
def bar_99():
        lst = [j for j in range(99)]
        s = sum(lst)
        return s
class Helper100:
def __init__(self, val): self.val = val
self.lst = [val]*3
