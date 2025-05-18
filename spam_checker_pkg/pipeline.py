from typing import Tuple
from . import rules, preprocess
from .constants import RULE_THRESHOLD, ML_PROB_THRESHOLD

def decide(text: str, model) -> Tuple[bool,int,float]:
    r_score = rules.score(text)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba([preprocess.clean(text)])[0][1]
    else:  # spaCy model
        prob = model(preprocess.clean(text)).cats["spam"]
    spam = (r_score >= RULE_THRESHOLD) or (prob >= ML_PROB_THRESHOLD)
    return spam, r_score, prob
