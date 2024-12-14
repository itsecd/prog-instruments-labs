import os


def get_next(class_name: str):
    """
    Returns the relative path for the class object

    This function returns the relative path for the class object passed
    to the function
    _summary_

    Args:
        class_name (str): _description_

    Yields:
        _type_: _description_
    """
    path = os.path.join('dataset', class_name)
    class_names = os.listdir(path)
    class_names.append(None)
    for i in range(len(class_names)):
        if class_names[i] is not None:
            yield os.path.join(path, class_names[i])
        elif class_names[i] is None:
            yield None


print(*get_next('leopard'))
