from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from .constants import DATA_DIR, RANDOM_SEED
from .train_ratio import make_80_20  # local helper

def _to_xy(df):
    X = df["text"]
    y = df["label"]
    return X, y

def load_local() -> pd.DataFrame:
    files = sorted(DATA_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError("Place your five CSVs in the 'data' folder.")
    dfs = [pd.read_csv(f, usecols=["CONTENT", "CLASS"]) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    return df.rename(columns={"CONTENT": "text", "CLASS": "label"})

def split_basic(df, test_size=0.25):
    X, y = _to_xy(df)
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_SEED
    )

def split_80_20(df, test_size=0.25):
    df_bal = make_80_20(df)
    X, y = _to_xy(df_bal)
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_SEED
    )

def split_20_80(df, test_size=0.25):
    spam = df[df.label == 1]
    ham  = df[df.label == 0]
    target_ham = int(len(spam) * 0.20 / 0.80)
    ham_sample = ham.sample(n=target_ham, random_state=RANDOM_SEED)
    new_df = pd.concat([ham_sample, spam]).sample(frac=1, random_state=RANDOM_SEED)
    X, y = _to_xy(new_df)
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_SEED
    )
