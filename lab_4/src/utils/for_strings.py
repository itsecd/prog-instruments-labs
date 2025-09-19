def limit_string(string: str, limit: int = 20) -> str:
    string_len = len(string)

    if string_len > limit:
        return string[:limit - 3] + "..."

    return string
