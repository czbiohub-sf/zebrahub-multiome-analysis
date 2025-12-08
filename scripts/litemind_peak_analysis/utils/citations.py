import re

# Pre-compiled regexes for speed and clarity
_PLACEHOLDER_PATTERNS = [
    re.compile(r"\[turn\d+[A-Za-z]+\d+\]"),                 # [turn1search0]
    re.compile(r"\[oaicite:\d+\]"),                         # [oaicite:0]
    re.compile(r"::?contentReference"),                     # ::contentReference or :contentReference
    re.compile(r"\[(?:REF|REFERENCE|SOURCE|CITATION NEEDED)\]", re.I),
]

def has_broken_citations(text: str) -> bool:
    """
    Return True if `text` contains any known citation placeholders or
    mismatched rich-text citation markers; otherwise False.
    """
    # 1️⃣ Direct regex matches
    if any(p.search(text) for p in _PLACEHOLDER_PATTERNS):
        return True

    return False