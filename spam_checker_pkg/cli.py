import argparse
import json
import sys
import random
import shutil
from pathlib import Path

import pandas as pd
from sklearn import metrics

from . import dataset, ml, spacy_model, pipeline
from .constants import MODEL_DIR, RANDOM_SEED


def _train_models(X_train, y_train):
    # scikit-learn models
    logreg = ml.train_model(ml.build_logreg(), X_train, y_train)
    svm    = ml.train_model(ml.build_svm(),    X_train, y_train)
    # spaCy model
    nlp_sp = spacy_model.train_spacy(
        spacy_model.build_spacy(), X_train, y_train, n_iter=10, drop=0.2
    )
    return {"logreg": logreg, "svm": svm, "spacy": nlp_sp}


def _evaluate_all(models, X_test, y_test):
    return [
        ml.evaluate_model("logreg", models["logreg"], X_test, y_test),
        ml.evaluate_model("svm",    models["svm"],    X_test, y_test),
        spacy_model.evaluate_model(models["spacy"],  X_test, y_test),
    ]


def main(argv=None):
    p = argparse.ArgumentParser(description="Spam-checker CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Train
    t = sub.add_parser("train", help="Train models")
    t.add_argument(
        "--variant", choices=["basic", "80-20", "20-80"], required=True,
        help="Data split variant"
    )

    # Evaluate
    e = sub.add_parser("evaluate", help="Evaluate saved models")
    e.add_argument(
        "--model", choices=["logreg", "svm", "spacy"],
        help="Which model to link (default: auto-pick best by F1)"
    )
    e.add_argument(
        "--variant", choices=["basic", "80-20", "20-80"],
        help="Specify variant when using --model (if multiple variants exist)"
    )

    # Predict
    pr = sub.add_parser("predict", help="Predict one text")
    pr.add_argument("text", help="Text to classify")

    

    args = p.parse_args(argv)

    if args.cmd == "train":
        # 1) Load and split data
        df = dataset.load_local()
        split_fn = {
            "basic": dataset.split_basic,
            "80-20": dataset.split_80_20,
            "20-80": dataset.split_20_80,
        }[args.variant]
        X_train, X_test, y_train, y_test = split_fn(df)

        # 2) Train all three
        models = _train_models(X_train, y_train)

        # 3) Detailed reports
        for name, model in models.items():
            print(f"\n=== {name.upper()} Classification Report ===")
            if hasattr(model, "predict"):
                y_pred = model.predict(X_test)
            else:
                probs = spacy_model.predict_proba(model, X_test)
                y_pred = [int(p >= 0.5) for p in probs]
            print(metrics.classification_report(y_test, y_pred, digits=4))

        # 4) Save models + summary
        MODEL_DIR.mkdir(exist_ok=True)
        ml.save(models["logreg"], f"{args.variant}_logreg.joblib")
        ml.save(models["svm"],    f"{args.variant}_svm.joblib")
        spacy_model.save(models["spacy"], f"{args.variant}_spacy")

        summary = _evaluate_all(models, X_test, y_test)
        (MODEL_DIR / f"{args.variant}_metrics.json").write_text(
            json.dumps(summary, indent=2)
        )
        print(f"\nSummary F1 scores:\n{json.dumps(summary, indent=2)}")

    elif args.cmd == "evaluate":
        metrics_files = list(MODEL_DIR.glob("*_metrics.json"))
        if not metrics_files:
            sys.exit("No metrics found. Run 'train' first.")

        # Collect records
        all_res = []
        for jf in metrics_files:
            data = json.loads(jf.read_text())
            variant = jf.stem.replace("_metrics", "")
            for entry in data:
                entry["variant"] = variant
                all_res.append(entry)

        # Manual or automatic(by f1) selection
        chosen = None
        if args.model:

            cands = [e for e in all_res if e["model"] == args.model
                     and (not args.variant or e["variant"] == args.variant)]
            if not cands:
                sys.exit(f"No metrics found for model='{args.model}'"
                         + (f" and variant='{args.variant}'" if args.variant else ""))

            if len(cands) > 1 and not args.variant:
                variants = sorted({e["variant"] for e in cands})
                sys.exit(f"Multiple variants for model='{args.model}': {variants}. "
                         "Specify --variant.")
            chosen = cands[0]
        else:
            # Autopick
            chosen = max(all_res, key=lambda x: x["f1"])

        print("Chosen model:", chosen)

        # Symlink creation
        dst = MODEL_DIR / "active"
        if dst.exists() or dst.is_symlink():
            if dst.is_symlink() or dst.is_file():
                dst.unlink()
            else:
                shutil.rmtree(dst)

        base = f"{chosen['variant']}_{chosen['model']}"
        path_joblib = MODEL_DIR / f"{base}.joblib"
        path_spacy  = MODEL_DIR / base

        if path_joblib.exists():
            dst.symlink_to(path_joblib.resolve())
        elif path_spacy.exists():
            dst.symlink_to(path_spacy.resolve())
        else:
            sys.exit(f"Model files not found for '{base}'")

        print("Linked model at artifacts/active")

    elif args.cmd == "predict":
        import joblib
        import spacy

        active = MODEL_DIR / "active"
        if not active.exists():
            sys.exit("Run 'evaluate' first to create artifacts/active")

        # Load
        if active.is_symlink() or active.suffix == ".joblib":
            model = joblib.load(active)
        else:
            model = spacy.load(active)

        spam, r_score, prob = pipeline.decide(args.text, model)
        print(json.dumps({
            "spam": bool(spam),
            "rule_score": r_score,
            "probability": prob
        }, indent=2))

if __name__ == "__main__":
    main()
