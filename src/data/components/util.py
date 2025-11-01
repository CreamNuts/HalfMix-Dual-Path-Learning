import re


def get_num(s: str) -> int:
    """
    Get the number in a string.
    Example:
    >>> get_num('aaa1301') -> 1301
    >>> get_num('aaa13_12') -> 13
    """
    m = re.search(r"\d+", s)
    return int(m.group())
