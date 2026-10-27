
import pandas as pd
import re
import unicodedata

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load the dataset
df = pd.read_csv("imh_complaints.csv")


# 2. Preprocess complaints
def preprocess_text(text):
    text = str(text)
    text = unicodedata.normalize("NFKC", text)

    # Replace URLs
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)

    # Redact CNIC-like numbers
    text = re.sub(r"(?<!\w)\d{5}-\d{7}-\d(?!\w)", " PII ", text)

    # Redact common Pakistani mobile-number formats
    text = re.sub(
        r"(?<!\w)(?:\+?92[- ]?)?0?3\d{2}[- ]?\d{7}(?!\w)",
        " PII ",
        text
    )

    text = text.lower()
    text = re.sub(r"[^\w\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


df["cleaned_text"] = df["complaint_text"].apply(preprocess_text)


# 3. Bag-of-Words representation
bow_vectorizer = CountVectorizer(ngram_range=(1, 2))
bow_matrix = bow_vectorizer.fit_transform(df["cleaned_text"])

print("\n--- BAG-OF-WORDS ---")
print("Matrix shape:", bow_matrix.shape)
print("Vocabulary size:", len(bow_vectorizer.vocabulary_))

bow_total = bow_matrix.shape[0] * bow_matrix.shape[1]
bow_sparsity = 100 * (1 - bow_matrix.nnz / bow_total)
print(f"Sparsity: {bow_sparsity:.2f}%")


# 4. TF-IDF representation
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf_matrix = tfidf_vectorizer.fit_transform(df["cleaned_text"])

print("\n--- TF-IDF ---")
print("Matrix shape:", tfidf_matrix.shape)
print("Vocabulary size:", len(tfidf_vectorizer.vocabulary_))

tfidf_total = tfidf_matrix.shape[0] * tfidf_matrix.shape[1]
tfidf_sparsity = 100 * (1 - tfidf_matrix.nnz / tfidf_total)
print(f"Sparsity: {tfidf_sparsity:.2f}%")


# 5. Top 10 TF-IDF terms for one Electricity and one Water ticket
def show_top_terms(ticket_id):
    row_index = df.index[df["ticket_id"] == ticket_id][0]
    scores = tfidf_matrix[row_index].toarray().flatten()
    terms = tfidf_vectorizer.get_feature_names_out()

    top_indices = scores.argsort()[-10:][::-1]

    print(f"\nTop TF-IDF terms for {ticket_id}:")
    for index in top_indices:
        if scores[index] > 0:
            print(f"{terms[index]}: {scores[index]:.4f}")


electricity_ticket = df.loc[
    df["department"] == "Electricity", "ticket_id"
].iloc[0]

water_ticket = df.loc[
    df["department"] == "Water", "ticket_id"
].iloc[0]

show_top_terms(electricity_ticket)
show_top_terms(water_ticket)


# 6. Cosine similarity: same department
same_pair = df[df["department"] == "Electricity"].iloc[:2]
same_indices = same_pair.index.tolist()

same_similarity = cosine_similarity(
    tfidf_matrix[same_indices[0]],
    tfidf_matrix[same_indices[1]]
)[0, 0]


# 7. Cosine similarity: different departments
different_pair = df[
    (df["department"] == "Electricity")
    | (df["department"] == "Water")
]

electricity_index = different_pair[
    different_pair["department"] == "Electricity"
].index[0]

water_index = different_pair[
    different_pair["department"] == "Water"
].index[0]

different_similarity = cosine_similarity(
    tfidf_matrix[electricity_index],
    tfidf_matrix[water_index]
)[0, 0]


print("\n--- COSINE SIMILARITY ---")
print(f"Same department (Electricity): {same_similarity:.4f}")
print(f"Different departments (Electricity vs Water): {different_similarity:.4f}")

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["cleaned_text"],
    df["department"],
    test_size=0.30,
    random_state=42,
    stratify=df["department"]
)

model_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train = model_vectorizer.fit_transform(X_train_text)
X_test = model_vectorizer.transform(X_test_text)

models = {
    "Multinomial Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC(random_state=42)
}

os.makedirs("figures", exist_ok=True)

results = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="weighted", zero_division=0
    )

    results[model_name] = {
        "Accuracy": accuracy,
        "Macro Precision": precision,
        "Macro Recall": recall,
        "Macro F1": f1,
        "Weighted Precision": weighted_precision,
        "Weighted Recall": weighted_recall,
        "Weighted F1": weighted_f1
    }

    print(f"\n{'=' * 50}")
    print(model_name)
    print(f"{'=' * 50}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall: {recall:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print(f"Weighted Precision: {weighted_precision:.4f}")
    print(f"Weighted Recall: {weighted_recall:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test, predictions, labels=sorted(df["department"].unique()),
        zero_division=0
    ))

    labels = sorted(df["department"].unique())
    cm = confusion_matrix(y_test, predictions, labels=labels)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    display.plot(xticks_rotation=45)
    plt.title(model_name)
    plt.tight_layout()
    plt.savefig(f"figures/{model_name.lower().replace(' ', '_')}_confusion_matrix.png")

    print("\nMisclassified Complaints:")
    test_results = pd.DataFrame({
        "ticket_id": df.loc[X_test_text.index, "ticket_id"].values,
        "complaint_text": df.loc[X_test_text.index, "complaint_text"].values,
        "Actual": y_test.values,
        "Predicted": predictions
    })

    errors = test_results[test_results["Actual"] != test_results["Predicted"]]

    if errors.empty:
        print("No misclassified complaints in this test set.")
    else:
        print(errors.to_string(index=False))

print("\nModel Comparison")
print(pd.DataFrame(results).T.round(4))
plt.close("all")
