import pandas as pd
from .constants import RANDOM_SEED

TARGET_RATIO = 0.20  # spam fraction in ham-majority set

def make_80_20(df: pd.DataFrame) -> pd.DataFrame:
    ham  = df[df.label == 0]
    spam = df[df.label == 1]
    target_spam = int(len(ham) * TARGET_RATIO / (1 - TARGET_RATIO))
    spam_sampled = spam.sample(n=target_spam, random_state=RANDOM_SEED)
    new_df = pd.concat([ham, spam_sampled]).sample(frac=1, random_state=RANDOM_SEED)
    return new_df
