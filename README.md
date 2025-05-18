# Spam‑Checker

NLP Spam Checker

## Quick start

```bash
git clone <repo>   # or unzip this archive
cd spam_checker_repo
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 1. Train three models on default (basic) split
```bash
python -m spam_checker_pkg.cli train --variant basic
```

### 2. Evaluate saved metrics and pick the best model or pick active model manually
```bash
python -m spam_checker_pkg.cli evaluate
```
OR
```bash
python -m spam_checker_pkg.cli evaluate --model svm --variant basic
```

### 3. Predict
```bash
python -m spam_checker_pkg.cli predict "Earn €5000 from home, click here"
# spam(bool) | rule_score(float) | probability(float)
```

## Variants

* `--variant basic`   – original label distribution (~2 k rows)
* `--variant 80-20`   – 80 % ham, 20 % spam (training set)
* `--variant 20-80`   – 20 % ham, 80 % spam

## Directory layout

```
spam_checker_pkg/    package code
artifacts/           trained models + metrics (generated)
data/                your five CSVs (not tracked) / CSV structure: CONTENT, CLASS
```

