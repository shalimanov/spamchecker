# spam_checker_pkg/spacy_model.py

import spacy
import random
from spacy.util import minibatch, compounding
from spacy.training.example import Example
from sklearn.metrics import f1_score
from .constants import RANDOM_SEED, MODEL_DIR
from .preprocess import clean


def build_spacy(model_name: str = "en_core_web_lg", use_blank: bool = False):
    """
    Build a spaCy pipeline for text classification.
    
    If use_blank=True, starts from a blank 'en' pipeline (no lookup tables needed).
    Otherwise loads the pretrained pipeline (requires installing spacy-lookups-data
    to provide the lexeme_norm tables).
    """
    # Components to disable (they pull in lookup tables by default)
    disable_pipes = ["parser", "tagger", "attribute_ruler", "lemmatizer", "ner"]

    if use_blank:
        # Fully blank English pipeline—no external lookups required
        nlp = spacy.blank("en")
    else:
        # Pretrained pipeline—requires spacy-lookups-data for normalization tables
        nlp = spacy.load(model_name, disable=disable_pipes)

    # Ensure tok2vec is present
    if "tok2vec" not in nlp.pipe_names:
        nlp.add_pipe("tok2vec", first=True)

    # Add or retrieve the TextCategorizer
    if "textcat" in nlp.pipe_names:
        textcat = nlp.get_pipe("textcat")
    else:
        textcat = nlp.add_pipe("textcat", last=True)
        textcat.add_label("spam")
        textcat.add_label("ham")

    return nlp


def train_spacy(nlp, X_train, y_train, n_iter: int = 10, drop: float = 0.2):
    """
    Train the TextCategorizer with compounding minibatches and dropout.
    Prints loss per epoch.
    """
    # Prepare training examples
    train_examples = []
    for text, label in zip(X_train, y_train):
        doc = nlp.make_doc(clean(text))
        cats = {"spam": bool(label), "ham": not label}
        train_examples.append(Example.from_dict(doc, {"cats": cats}))

    # Initialize the model (providing examples to cache them)
    optimizer = nlp.initialize(lambda: train_examples)
    random.seed(RANDOM_SEED)

    # Training loop
    for epoch in range(n_iter):
        random.shuffle(train_examples)
        losses = {}
        batches = minibatch(
            train_examples,
            size=compounding(4.0, 32.0, 1.001)
        )
        for batch in batches:
            nlp.update(batch, sgd=optimizer, drop=drop, losses=losses)
        print(f"Epoch {epoch+1}/{n_iter} — Losses: {losses}")

    return nlp


def predict_proba(nlp, texts):
    """
    Return list of spam probabilities for the given texts.
    """
    probs = []
    for t in texts:
        doc = nlp(clean(t))
        probs.append(doc.cats.get("spam", 0.0))
    return probs


def evaluate_model(nlp, X_test, y_test):
    """
    Compute F1-score on the test set.
    """
    probs = predict_proba(nlp, X_test)
    y_pred = [int(p >= 0.5) for p in probs]
    f1 = f1_score(y_test, y_pred)
    return {"model": "spacy", "f1": f1}


def save(nlp, folder_name: str):
    """
    Save the spaCy model to artifacts/<folder_name>
    """
    dest = MODEL_DIR / folder_name
    dest.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(dest)


def load(folder_name: str):
    """
    Load a saved spaCy model from artifacts/<folder_name>
    """
    return spacy.load(MODEL_DIR / folder_name)
