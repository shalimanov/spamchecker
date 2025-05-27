import re
from functools import lru_cache
import spacy

_URL_RE = re.compile(r"http[s]?://\S+")
_NUM_RE = re.compile(r"\d+")

@lru_cache(maxsize=1)
def nlp():
    return spacy.load(
        "en_core_web_lg",
        disable=["parser", "ner"]
    )

def clean(text: str) -> str:
    text = text.lower()
    text = _URL_RE.sub("<URL>", text)
    text = _NUM_RE.sub("<NUM>", text)

    doc = nlp()(text)
    lemmas = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(lemmas)
