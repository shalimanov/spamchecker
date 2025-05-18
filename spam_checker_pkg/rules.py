import re
from rapidfuzz import fuzz, process
from .constants import SPAM_KEYWORDS, SPAM_TEMPLATES, KEYWORD_SCORE, TEMPLATE_SCORE

_KEYWORD_RES = [re.compile(p, flags=re.I) for p in SPAM_KEYWORDS]

def score(text: str) -> int:
    text_lc = text.lower()
    s = sum(bool(r.search(text_lc)) for r in _KEYWORD_RES) * KEYWORD_SCORE
    best = process.extractOne(text_lc, SPAM_TEMPLATES, scorer=fuzz.ratio, score_cutoff=80)
    if best:
        s += TEMPLATE_SCORE
    return s
