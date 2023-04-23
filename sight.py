import re
from typing import Tuple

TMP_ERICA_TEXT_FILE = "/tmp/erica_text.txt"

def get_context() -> str:
    with open(TMP_ERICA_TEXT_FILE, "r") as f:
        return f.read()

def does_contain_url(string: str) -> Tuple[bool, str]:
    """
    Check if a string contains a URL.

    Returns a tuple of a boolean and the URL if found.
    """
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.findall(regex, string)
    if not url:
        return False, ""
    else:
        return True, url[0][0]
