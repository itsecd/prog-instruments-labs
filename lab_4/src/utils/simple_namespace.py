from types import SimpleNamespace


def dict_to_sn(d: dict) -> SimpleNamespace:
    """
    Converts a dict to a SimpleNamespace

    Args:
        d (dict)

    Returns:
        SimpleNamespace
    """
    sn = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(sn, key, dict_to_sn(value))
        else:
            setattr(sn, key, value)

    return sn
