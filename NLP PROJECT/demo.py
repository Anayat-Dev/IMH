"""Interactive demo for the original 40-ticket NLP PROJECT dataset.
Run from this folder: python demo.py
Type a complaint and press Enter; type q to quit.
The department is a model suggestion and must be confirmed by a human.
"""
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC

BASE = Path(__file__).resolve().parent
DATA = BASE / "imh_complaints.csv"


def preprocess_text(text):
    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"(?<!\w)\d{5}-\d{7}-\d(?!\w)", " PII ", text)
    text = re.sub(r"(?<!\w)(?:\+?92[- ]?)?0?3\d{2}[- ]?\d{7}(?!\w)", " PII ", text)
    text = text.lower()
    text = re.sub(r"[^\w\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_model():
    df = pd.read_csv(DATA).dropna(subset=["complaint_text", "department"])
    X = df["complaint_text"].astype(str).map(preprocess_text)
    y = df["department"].astype(str)
    # Character n-grams help with spelling variation; word features preserve meaning.
    model = Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), sublinear_tf=True)),
        ])),
        ("classifier", LinearSVC(class_weight="balanced", random_state=42)),
    ])
    model.fit(X, y)
    return model, sorted(y.unique())


def route_obvious_complaint(complaint, model_prediction):
    """Correct only high-confidence, explicit service keywords in the demo."""
    text = preprocess_text(complaint)

    electricity_terms = (
        r"\b(electricity|electric|power outage|power cut|no power|bijli|iesco|"
        r"load shedding|transformer|voltage fluctuation|sparking wires|electric meter)\b"
    )
    water_terms = r"\b(water|tanker|drinking water|water supply|pipe burst|hand pump)\b"

    # If the complaint explicitly names electricity and not a water issue,
    # route it to Electricity; otherwise retain the trained model prediction.
    if re.search(electricity_terms, text) and not re.search(water_terms, text):
        return "Electricity"
    return model_prediction


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    model, departments = build_model()
    print("=" * 48)
    print(" IMH COMPLAINT ROUTING — NLP PROJECT DEMO")
    print("=" * 48)
    print("Departments:", ", ".join(departments))
    print("Enter a complaint, or type q to quit.")
    print("Note: predictions are suggestions; a human should confirm routing.\n")
    while True:
        try:
            complaint = input("Complaint> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nDemo closed.")
            break
        if complaint.lower() in {"q", "quit", "exit"}:
            print("Demo closed.")
            break
        if not complaint:
            print("Please enter a complaint.\n")
            continue
        cleaned = preprocess_text(complaint)
        prediction = model.predict([cleaned])[0]
        prediction = route_obvious_complaint(complaint, prediction)
        print("Cleaned text:", cleaned)
        print("Suggested department:", prediction)
        print("Human review: required, especially for Health or uncertain complaints.\n")


if __name__ == "__main__":
    main()
