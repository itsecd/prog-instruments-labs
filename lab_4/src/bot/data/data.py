import sqlite3
import json

db = r'C:\Users\user\Desktop\UNIVERSITY\prog-instruments-labs\lab_4\data\users.db'    
conn = sqlite3.connect(db)
cursor = conn.cursor()
print("database was connected")

cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    userid TEXT PRIMARY KEY,
    usertasks TEXT
)
''')

def getCurNumOfTask(userid):
    tasks = getUserTasks(userid)
    return (len(tasks) + 1)

def getUserTasks(userid):
    cursor.execute('SELECT usertasks FROM users WHERE userid = ?', (userid,))
    result = cursor.fetchone()
    if result and result[0] is not None:
        try:
            tasks = json.loads(result[0])
            if isinstance(tasks, list):
                return tasks
            else:
                return []
        except json.JSONDecodeError:
            print()
            return []
    else:
        return []

def taskAdder(datafu):
    try:
        currentTasks = getUserTasks(datafu['userid'])
        newtasks = [datafu['task']]
        if currentTasks == []:
            currentTasks = newtasks
        else:
            currentTasks.extend(newtasks)
            
        tasks_json = json.dumps(currentTasks)  
        cursor.execute('INSERT OR REPLACE INTO users (userid, usertasks) VALUES (?, ?)', (datafu['userid'], tasks_json))
        conn.commit()
        print("successful writing db")
        return True
    except Exception as e:
        print(f"error with db writing: {e}")
        return False

def clearTasks(userid):
    try:
        cursor.execute('UPDATE users SET usertasks = NULL WHERE userid = ?', (userid,))
        conn.commit()
        return True
    except Exception as e:
        print(f"error with db clearing: {e}")
        return False

def reverseTasks(userid):
    try:
        tasks = getUserTasks(userid)
        tasks.reverse()
        
        tasks_json = json.dumps(tasks)  
        cursor.execute('REPLACE INTO users (userid, usertasks) VALUES (?, ?)', (userid, tasks_json))
        conn.commit()
        print("successful writing db")
        return True
    except Exception as e:
        print(f"error with db reversing: {e}")
        return False

def editTasks(datafu):
    try:
        currentTasks = getUserTasks(datafu['userid'])
        tasknums = datafu['tasknums']
        newtasks = datafu['editedtasks']
        
        for i, el in enumerate(tasknums):
            currentTasks[int(el) - 1] = newtasks[i]
        
        tasks_json = json.dumps(currentTasks)  
        cursor.execute('REPLACE INTO users (userid, usertasks) VALUES (?, ?)', (datafu['userid'], tasks_json))
        conn.commit()
        print("successful writing db")
        return True
    except Exception as e:
        print(f"error with db writing: {e}")
        return False

def deleteTasks(datafu):
    try:
        currentTasks = getUserTasks(datafu['userid'])
        tasknums = datafu['tasknums']
        print(f"ct: {currentTasks}")
        print(sorted(tasknums, reverse=True))
        for index in sorted(tasknums, reverse=True):
            if index - 1 < len(currentTasks):
                print(f"i: {index}, {currentTasks[index - 1]}")
                currentTasks.pop(index - 1)                 
            
        tasks_json = json.dumps(currentTasks)  
        cursor.execute('INSERT OR REPLACE INTO users (userid, usertasks) VALUES (?, ?)', (datafu['userid'], tasks_json))
        conn.commit()
        print("successful writing db")
        return True
    except Exception as e:
        print(f"error with db writing: {e}")
        return False