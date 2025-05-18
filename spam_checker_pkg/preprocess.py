import re
from functools import lru_cache
import spacy

_URL_RE = re.compile(r"http[s]?://\S+")
_NUM_RE = re.compile(r"\d+")

@lru_cache(maxsize=1)
def nlp():
    return spacy.blank("en", disable=["parser", "ner", "lemmatizer"])

def clean(text: str) -> str:
    text = text.lower()
    text = _URL_RE.sub("<URL>", text)
    text = _NUM_RE.sub("<NUM>", text)
    return text.strip()
