"""Reproducible 5-fold stratified comparison: word TF-IDF vs word+character TF-IDF."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

BASE = Path(__file__).resolve().parent
df = pd.read_csv(BASE / "imh_complaints.csv").dropna(subset=["complaint_text", "department"])
X = df["complaint_text"].astype(str)
y = df["department"].astype(str)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = {
    "Model A - Word TF-IDF": Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("svm", LinearSVC(class_weight="balanced", random_state=42)),
    ]),
    "Model B - Word + Character TF-IDF": Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), sublinear_tf=True)),
        ])),
        ("svm", LinearSVC(class_weight="balanced", random_state=42)),
    ]),
}
scoring = {"accuracy": "accuracy", "macro_f1": "f1_macro", "weighted_f1": "f1_weighted"}
print("5-FOLD STRATIFIED CROSS-VALIDATION")
print("Dataset size:", len(df))
print("Department counts:\n", y.value_counts().to_string())
rows = []
for name, model in models.items():
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
    row = {"Model": name}
    print("\n" + "=" * 45 + "\n" + name)
    for metric in scoring:
        values = scores["test_" + metric]
        row[metric] = values.mean()
        row[metric + "_std"] = values.std()
        print(f"{metric}: {values.mean():.4f} +/- {values.std():.4f}")
        print("Fold scores:", [round(float(v), 3) for v in values])
    rows.append(row)
out = pd.DataFrame(rows)
out.to_csv(BASE / "results" / "cv_model_comparison.csv", index=False)
print("\nFINAL COMPARISON\n", out.round(4).to_string(index=False))
print("\nSaved results/cv_model_comparison.csv")
print("Note: only 40 examples; treat these estimates as preliminary.")
