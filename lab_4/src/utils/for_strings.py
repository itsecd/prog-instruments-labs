def limit_string(string: str, limit: int = 20) -> str:
    """
    Truncates a string to the required number of characters.

    Args:
        string (str): original string
        limit (int, optional): output string length. Defaults to 20.

    Returns:
        str: truncated string
    """
    string_len = len(string)

    if string_len > limit:
        return string[:limit - 3] + "..."

    return string
