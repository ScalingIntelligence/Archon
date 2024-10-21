import re
from typing import List, Set


def _split_by_any_char(input_string: str, delimiters: Set[str]) -> List[str]:
    """Split a string by any of the given delimiter characters.

    Args:
        input_string (str): The string to be split.
        delimiters (set[str]): A set of single-character strings to use as delimiters.

    Returns:
        list[str]: A list of substrings resulting from splitting the input string.

    Raises:
        ValueError: If delimiters is empty or if any delimiter is not a single-character string.
    """
    if not delimiters:
        raise ValueError("Delimiters set cannot be empty")

    if any(len(delimiter) != 1 for delimiter in delimiters):
        raise ValueError("All delimiters must be single-character strings")

    # Escape special regex characters and join delimiters
    pattern = "|".join(re.escape(delimiter) for delimiter in delimiters)

    # Split the input string using the pattern
    return re.split(pattern, input_string)


# Delimiters per https://github.com/google-deepmind/code_contests/blob/fa7a4f8139aab08362503f3344778eb86901709a/execution/tester_sandboxer.cc#L137C43-L137C44
DELIMITERS = {" ", "\n", "\t", "\r", "\v"}


def _split_and_lowercase(input_string: str) -> str:
    """Splits a string on delimiters and converts to lowercase.

    Mimics this function:
    https://github.com/google-deepmind/code_contests/blob/fa7a4f8139aab08362503f3344778eb86901709a/execution/tester_sandboxer.cc#L135
    """
    parts = _split_by_any_char(input_string, DELIMITERS)

    parts_lowered_and_filtered = [part.lower() for part in parts if len(part)]
    return parts_lowered_and_filtered


def _parse_to_value(string: int) -> str | int | float:
    try:
        return int(string)
    except ValueError:
        pass

    try:
        return float(string)
    except ValueError:
        return string


# From here https://github.com/google-deepmind/code_contests/blob/fa7a4f8139aab08362503f3344778eb86901709a/execution/tester_sandboxer.cc#L146C6-L146C17
K_DOUBLE_PRECISION = 1e-5


def _values_match(a: str, b: str) -> bool:
    """ "
    Mimics this function https://github.com/google-deepmind/code_contests/blob/fa7a4f8139aab08362503f3344778eb86901709a/execution/tester_sandboxer.cc#L146C6-L146C17
    """
    a_value = _parse_to_value(a)
    b_value = _parse_to_value(b)

    if isinstance(a_value, str) or isinstance(b_value, str):
        # If either are only a string, do string comparison
        return a == b
    elif isinstance(a_value, float) or isinstance(b_value, float):
        # If the numbers are numeric and one is a float, do a float comparison
        return abs(a_value - b_value) < K_DOUBLE_PRECISION
    elif isinstance(a_value, int) and isinstance(b_value, int):
        return a_value == b_value
    else:
        raise AssertionError("Invalid case.")


def outputs_match(output_a: str, output_b: str) -> bool:
    a_parts = _split_and_lowercase(output_a)
    b_parts = _split_and_lowercase(output_b)

    if len(a_parts) != len(b_parts):
        return False

    for a_part, b_part in zip(a_parts, b_parts):
        if not _values_match(a_part, b_part):
            return False

    return True
