"""
Fetches 20 Newsgroups (real, via scikit-learn - needs internet on first run),
TF-IDF vectorizes the post text, trains three classifiers, evaluates each on
the held-out test split, and writes every artifact the dashboard needs:

    vectorizer.pkl              fitted TfidfVectorizer
    model_<name>.pkl            one per model in config.MODEL_NAMES
    results.json                accuracy/F1/confusion matrix per model
    test_predictions.csv        every test post + true label + each model's
                                 prediction, for the dashboard's error-analysis
                                 view (avoids re-running inference at dashboard
                                 load time)

Run: python train.py
"""
import json

import joblib
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from config import (
    CATEGORIES, MODEL_NAMES, MODEL_PATH_TMPL, RESULTS_PATH,
    TEST_DATA_PATH, VECTORIZER_PATH,
)

REMOVE = ("headers", "footers", "quotes")  # strip metadata so models learn content, not headers


def build_model(name: str):
    if name == "logistic_regression":
        return LogisticRegression(max_iter=1000)
    if name == "naive_bayes":
        return MultinomialNB()
    if name == "linear_svm":
        return LinearSVC()
    raise ValueError(f"unknown model name: {name}")


def main():
    print(f"Fetching 20 Newsgroups, categories={CATEGORIES} ...")
    train_raw = fetch_20newsgroups(subset="train", categories=CATEGORIES, remove=REMOVE, random_state=42)
    test_raw = fetch_20newsgroups(subset="test", categories=CATEGORIES, remove=REMOVE, random_state=42)
    target_names = train_raw.target_names
    print(f"Train posts: {len(train_raw.data)}  Test posts: {len(test_raw.data)}")

    vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
    X_train = vectorizer.fit_transform(train_raw.data)
    X_test = vectorizer.transform(test_raw.data)
    y_train, y_test = train_raw.target, test_raw.target
    joblib.dump(vectorizer, VECTORIZER_PATH)

    results = {}
    pred_df = pd.DataFrame({
        "text": test_raw.data,
        "true_label": [target_names[i] for i in y_test],
    })

    for name in MODEL_NAMES:
        print(f"Training {name} ...")
        model = build_model(name)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, target_names=target_names, output_dict=True)
        cm = confusion_matrix(y_test, preds).tolist()

        results[name] = {
            "accuracy": acc,
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_f1": report["weighted avg"]["f1-score"],
            "per_class": {
                cls: {"precision": report[cls]["precision"], "recall": report[cls]["recall"], "f1": report[cls]["f1-score"]}
                for cls in target_names
            },
            "confusion_matrix": cm,
            "labels": target_names,
        }
        pred_df[f"pred_{name}"] = [target_names[i] for i in preds]

        joblib.dump(model, MODEL_PATH_TMPL.format(name=name))
        print(f"  {name}: accuracy={acc:.3f}  macro_f1={report['macro avg']['f1-score']:.3f}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    pred_df.to_csv(TEST_DATA_PATH, index=False)

    best = max(results, key=lambda n: results[n]["macro_f1"])
    print(f"\nBest model by macro F1: {best} ({results[best]['macro_f1']:.3f})")
    print(f"Saved {VECTORIZER_PATH}, one model_*.pkl per model, {RESULTS_PATH}, {TEST_DATA_PATH}")


if __name__ == "__main__":
    main()
