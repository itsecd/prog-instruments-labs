import sys
import time
import copy
'''
Example of test.txt file (double walls around the 8x8 or ?x? grid is required):
WWWWWWWWWWWW
WWWWWWWWWWWW
WW........WW
WW........WW
WW........WW
WW.WR..PW.WW
WW...W.W..WW
WW........WW
WW...S....WW
WW........WW
WWWWWWWWWWWW
WWWWWWWWWWWW
'''


# -------------------------------------------------------------------------------
def print_history(history):
    print('')
    for h in history:
        print('Action: {} / Cost: {}'.format(h[1], h[2]))
        state = h[0]
        loc = h[3]
        state[loc[0]][loc[1]] = '*'
        for m in h[0]:
            print(''.join(m))
        print('')
        state[loc[0]][loc[1]] = '.'
    print('------------------------------------------')


# -------------------------------------------------------------------------------
def solve(history):
    """Recursive function that solves the board.
    history: list of lists
        [0] state: list of lists that is the board matrix
        [1] action: string
        [2] cost: integer
        [3] current location list: [r, c]
    """
    global cost_best
    global history_best
    global bails

    cost = history[-1][2]

    # print_history(history)
    # raw_input('Continue')

    # Scan for remaining monster health
    health_left = monster_health_remaining(history)
    if not health_left:
        if cost_best > cost:
            cost_best = cost
            history_best = history

            # print_history(history)
            # raw_input('Continue')

            print('Best cost:', cost_best)
        return

    # Optimization: Bail if minimum future cost already beaten
    if cost + minimum_extra_cost(health_left) >= cost_best:
        bails += 1
        if bails % 10000 == 0:
            print('bails:', bails)
        # print_history(history)
        # raw_input('Continue')
        return

    # Get valid floors that are reachable
    loc = history[-1][3]
    r = loc[0]
    c = loc[1]
    walked = [[r, c]]
    valid_floors = [[r, c]]
    get_valid_floors(history, walked, valid_floors)

    # Complete all swords first; seems to work faster
    for loc in valid_floors:
        r = loc[0]
        c = loc[1]
        check_sword_attacks(history, r, c)

    for loc in valid_floors:
        r = loc[0]
        c = loc[1]
        check_spear_attacks(history, r, c)

    for loc in valid_floors:
        r = loc[0]
        c = loc[1]
        check_dagger_attacks(history, r, c)

    for loc in valid_floors:
        r = loc[0]
        c = loc[1]
        check_bow_attacks(history, r, c)


# -------------------------------------------------------------------------------
def get_valid_floors(history, walked, valid_floors):
    state = history[-1][0]
    loc = walked[-1]
    r = loc[0]
    c = loc[1]

    # Try north
    if state[r - 1][c] == floor and [r - 1, c] not in valid_floors:
        walked_new = copy.deepcopy(walked)
        walked_new.append([r - 1, c])
        valid_floors.append([r - 1, c])
        get_valid_floors(history, walked_new, valid_floors)

    # Try east
    if state[r][c + 1] == floor and [r, c + 1] not in valid_floors:
        walked_new = copy.deepcopy(walked)
        walked_new.append([r, c + 1])
        valid_floors.append([r, c + 1])
        get_valid_floors(history, walked_new, valid_floors)

    # Try south
    if state[r + 1][c] == floor and [r + 1, c] not in valid_floors:
        walked_new = copy.deepcopy(walked)
        walked_new.append([r + 1, c])
        valid_floors.append([r + 1, c])
        get_valid_floors(history, walked_new, valid_floors)

    # Try west
    if state[r][c - 1] == floor and [r, c - 1] not in valid_floors:
        walked_new = copy.deepcopy(walked)
        walked_new.append([r, c - 1])
        valid_floors.append([r, c - 1])
        get_valid_floors(history, walked_new, valid_floors)


# ------------------------------------------------------------------------------
def minimum_extra_cost(health_left):
    """Extremely important optimization to reduce recursion massively."""
    extra_cost = (health_left // 3) * 80
    health_left %= 3
    if health_left == 2:
        extra_cost += 70
    if health_left == 1:
        extra_cost += 50
    return extra_cost


# -------------------------------------------------------------------------------
def monster_health_remaining(history):
    """Calculates the remaining health of monsters on the board."""
    health_left = 0
    state = history[-1][0]
    for r in range(len(state)):
        for c in range(len(state[0])):
            if state[r][c] == monster_red:
                health_left += 3
            if state[r][c] == monster_purple:
                health_left += 2
            if state[r][c] == monster_blue:
                health_left += 1
    return health_left


# -------------------------------------------------------------------------------
def check_sword_attacks(history, r, c):
    # Only swing if two or three monsters hit or if one diagonal
    # without space for dagger swing
    # and potential push (results in same effect for cheaper)
    state = history[-1][0]
    for direction in 'NESW':
        attacks = []
        do_attack = False
        monster_count = 0

        # Check north
        if direction == 'N':
            for c_offset in range(-1, 2):
                if state[r - 1][c + c_offset] in monsters:
                    monster_count += 1
            if monster_count > 1:
                do_attack = True
            elif monster_count == 1:
                if state[r - 1][c - 1] in monsters and \
                        state[r][c - 1] != floor:
                    do_attack = True
                if state[r - 1][c + 1] in monsters and \
                        state[r][c + 1] != floor:
                    do_attack = True
            if do_attack:
                attacks = [[r - 1, c - 1], [r - 1, c], [r - 1, c + 1]]
                push_dir_col = 0
                push_dir_row = -1

        # Check east
        if direction == 'E':
            for r_offset in range(-1, 2):
                if state[r + r_offset][c + 1] in monsters:
                    monster_count += 1
            if monster_count > 1:
                do_attack = True
            elif monster_count == 1:
                if state[r - 1][c + 1] in monsters and \
                        state[r - 1][c] != floor:
                    do_attack = True
                if state[r + 1][c + 1] in monsters and \
                        state[r + 1][c] != floor:
                    do_attack = True
            if do_attack:
                attacks = [[r - 1, c + 1], [r, c + 1], [r + 1, c + 1]]
                push_dir_col = 1
                push_dir_row = 0

        # Check south
        if direction == 'S':
            for c_offset in range(-1, 2):
                if state[r + 1][c + c_offset] in monsters:
                    monster_count += 1
            if monster_count > 1:
                do_attack = True
            elif monster_count == 1:
                if state[r + 1][c - 1] in monsters and \
                        state[r][c - 1] != floor:
                    do_attack = True
                if state[r + 1][c + 1] in monsters and \
                        state[r][c + 1] != floor:
                    do_attack = True
            if do_attack:
                attacks = [[r + 1, c - 1], [r + 1, c], [r + 1, c + 1]]
                push_dir_col = 0
                push_dir_row = 1

        # Check west
        if direction == 'W':
            for r_offset in range(-1, 2):
                if state[r + r_offset][c - 1] in monsters:
                    monster_count += 1
            if monster_count > 1:
                do_attack = True
            elif monster_count == 1:
                if state[r - 1][c - 1] in monsters and \
                        state[r - 1][c] != floor:
                    do_attack = True
                if state[r + 1][c - 1] in monsters and \
                        state[r + 1][c] != floor:
                    do_attack = True
            if do_attack:
                attacks = [[r - 1, c - 1], [r, c - 1], [r + 1, c - 1]]
                push_dir_col = -1
                push_dir_row = 0

        if len(attacks):
            do_attacks(history, attacks, push_dir_row, push_dir_col,
                       'Sword', 80, r, c)


# -------------------------------------------------------------------------------
def check_spear_attacks(history, r, c):
    # There's no point in using a spear attack unless it's hitting 2 monsters.
    # If only one next to knight, dagger is cheaper.
    # If only one two away from knight, bow is cheaper.
    state = history[-1][0]
    for direction in 'NESW':
        attacks = []

        # Check north
        if direction == 'N':
            if state[r - 2][c] in monsters and \
                    state[r - 1][c] in monsters:
                attacks = [[r - 2, c], [r - 1, c]]
                push_dir_col = 0
                push_dir_row = -1

        # Check east
        if direction == 'E':
            if state[r][c + 2] in monsters and \
                    state[r][c + 1] in monsters:
                attacks = [[r, c + 2], [r, c + 1]]
                push_dir_col = 1
                push_dir_row = 0

        # Check south
        if direction == 'S':
            if state[r + 2][c] in monsters and \
                    state[r + 1][c] in monsters:
                attacks = [[r + 2, c], [r + 1, c]]
                push_dir_col = 0
                push_dir_row = 1

        # Check west
        if direction == 'W':
            if state[r][c - 2] in monsters and \
                    state[r][c - 1] in monsters:
                attacks = [[r, c - 2], [r, c - 1]]
                push_dir_col = -1
                push_dir_row = 0

        if len(attacks):
            do_attacks(history, attacks, push_dir_row, push_dir_col,
                       'Spear', 70, r, c)


# -------------------------------------------------------------------------------
def check_bow_attacks(history, r, c):
    """Checks for valid bow attacks."""
    state = history[-1][0]
    for direction in 'NESW':
        attacks = []

        # Check north
        if direction == 'N':
            if state[r - 2][c] in monsters and \
                    state[r - 1][c] != floor:
                attacks = [[r - 2, c]]
                push_dir_col = 0
                push_dir_row = -1

        # Check east
        if direction == 'E':
            if state[r][c + 2] in monsters and \
                    state[r][c + 1] != floor:
                attacks = [[r, c + 2]]
                push_dir_col = 1
                push_dir_row = 0

        # Check south
        if direction == 'S':
            if state[r + 2][c] in monsters and \
                    state[r + 1][c] != floor:
                attacks = [[r + 2, c]]
                push_dir_col = 0
                push_dir_row = 1

        # Check west
        if direction == 'W':
            if state[r][c - 2] in monsters and \
                    state[r][c - 1] != floor:
                attacks = [[r, c - 2]]
                push_dir_col = -1
                push_dir_row = 0

        if len(attacks):
            do_attacks(history, attacks, push_dir_row, push_dir_col,
                       'Bow', 60, r, c)


# -------------------------------------------------------------------------------
def check_dagger_attacks(history, r, c):
    """Checks for valid dagger attacks."""
    state = history[-1][0]
    for direction in 'NESW':
        attacks = []

        # Check north
        if direction == 'N':
            if state[r - 1][c] in monsters:
                attacks = [[r - 1, c]]
                push_dir_col = 0
                push_dir_row = -1

        # Check east
        if direction == 'E':
            if state[r][c + 1] in monsters:
                attacks = [[r, c + 1]]
                push_dir_col = 1
                push_dir_row = 0

        # Check south
        if direction == 'S':
            if state[r + 1][c] in monsters:
                attacks = [[r + 1, c]]
                push_dir_col = 0
                push_dir_row = 1

        # Check west
        if direction == 'W':
            if state[r][c - 1] in monsters:
                attacks = [[r, c - 1]]
                push_dir_col = -1
                push_dir_row = 0

        if len(attacks):
            do_attacks(history, attacks, push_dir_row, push_dir_col,
                       'Dagger', 50, r, c)


# -------------------------------------------------------------------------------
def do_attacks(history, attacks, push_dir_row,
               push_dir_col, action, cost_add, r, c):
    """Executes the attack based on the action and updates the history."""
    global history_complete

    history = copy.deepcopy(history)
    state = copy.deepcopy(history[-1][0])
    cost = history[-1][2]
    for a in attacks:
        if state[a[0]][a[1]] in monsters:
            mon = hit_monster(state[a[0]][a[1]])
            if state[a[0] + push_dir_row][a[1] + push_dir_col] == floor:
                state[a[0] + push_dir_row][a[1] + push_dir_col] = mon
                state[a[0]][a[1]] = floor
            else:
                state[a[0]][a[1]] = mon
    action += ' ' + do_text_direction(push_dir_row, push_dir_col)
    cost += cost_add
    """
    do_append = True
    for item in history_complete:
        if item[0] == state and item[1] < cost:
            do_append = False
            break

    if do_append:
        history_complete.append([state, cost])
        history.append([state, action, cost, [r, c]])

        solve(history)
    """
    # This is way faster than the above even though it is recording
    # more data than necessary
    if [state, cost] not in history_complete:
        history_complete.append([state, cost])
        history.append([state, action, cost, [r, c]])
        solve(history)


# -------------------------------------------------------------------------------
def do_text_direction(push_dir_row, push_dir_col):
    """Returns the direction as a string based on row and column offsets."""
    if push_dir_row == -1:
        return 'North'
    if push_dir_row == 1:
        return 'South'
    if push_dir_col == -1:
        return 'West'
    if push_dir_col == 1:
        return 'East'


# -------------------------------------------------------------------------------
def hit_monster(mon):
    """Returns the next rank of the monster after being hit."""
    mon_ranks = [monster_red, monster_purple, monster_blue, floor]
    return mon_ranks[mon_ranks.index(mon) + 1]


# -------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------
time_start = time.time()

monsters = 'RPB'
wall = 'W'
floor = '.'
monster_blue = 'B'
monster_purple = 'P'
monster_red = 'R'
start = 'S'

# Optimization:
# history_complete is used to ignore duplicate state and cost situations,
# which is common in recursive bow and dagger attacks;
# only includes state[0] and cost[1]
history_complete = []

history_best = []
cost_best = 9999
bails = 0

if len(sys.argv) > 1:
    filename = sys.argv[1][1:]
else:
    filename = 'test.txt'
    print('You can use \'python solver.py -[filename]\'')
print('Reading filename: ', filename)
with open(filename) as f:
    state = f.read().split('\n')

# Convert strings to lists
for i in range(len(state)):
    state[i] = list(state[i])

# Find the start
loc_start = [-1, -1]
for r in range(len(state)):
    for c in range(len(state[0])):
        if state[r][c] == start:
            loc_start[0], loc_start[1] = r, c
            state[r][c] = floor
if loc_start[0] == -1:
    print('No start found (\'S\' on board)')
    exit()

solve([[state, 'Start', 0, loc_start]])

# Output best option found
print('------------------------------------------')
print('------------------------------------------')
print('------------------------------------------')
print('')
print('Best Cost:', cost_best)
print_history(history_best)

print('Time elapsed:', time.time() - time_start)
input('Continue')
