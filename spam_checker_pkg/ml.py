from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn import metrics
import joblib
from .constants import RANDOM_SEED, MODEL_DIR
from .preprocess import clean

def _tfidf():
    return TfidfVectorizer(
        lowercase=False,  # already cleaned
        ngram_range=(1,2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True
    )

def build_logreg():
    logreg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED)
    return Pipeline([("tfidf", _tfidf()), ("clf", logreg)])

def build_svm():
    svc = LinearSVC(class_weight="balanced", random_state=RANDOM_SEED)
    cal = CalibratedClassifierCV(svc)
    return Pipeline([("tfidf", _tfidf()), ("clf", cal)])

def train_model(p: Pipeline, X, y):
    p.fit(X, y)
    return p

def evaluate_model(name: str, p: Pipeline, X, y):
    y_pred = p.predict(X)
    f1 = metrics.f1_score(y, y_pred)
    return { "model": name, "f1": f1 }

def save(p: Pipeline, fname: str):
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(p, MODEL_DIR / fname)

def load(fname: str) -> Pipeline:
    return joblib.load(MODEL_DIR / fname)
