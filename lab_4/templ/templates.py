def RenderTaskList(tasks):
    result = ""
    for i, el in enumerate(tasks):
        result += f"{i+1}.) {el}\n"
    return result
