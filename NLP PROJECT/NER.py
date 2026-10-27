
import pandas as pd
import spacy
import re

df = pd.read_csv("imh_complaints.csv")
nlp = spacy.load("en_core_web_sm")

locations = [
    "G-9", "F-11 Markaz", "F-11", "I-8 Markaz", "I-8",
    "G-11", "BHU Koral", "Bhara Kahu", "Diplomatic Enclave",
    "Service Road South", "Karachi Company", "Islamabad",
    "Centaurus Mall", "Faisal Mosque", "Street 12", "Street 7"
]

organizations = [
    "IESCO", "CDA", "PIMS", "IMH", "Traffic Police",
    "Capital Development Authority", "BHU"
]

def extract_entities(text):
    doc = nlp(text)
    entities = []

    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "FAC"]:
            label = ent.label_

            if label == "PERSON":
                label = "PER"
            elif label in ["GPE", "LOC", "FAC"]:
                label = "LOC"

            if label == "ORG" and any(
                re.search(r"(?<!\w)" + re.escape(location) + r"(?!\w)", ent.text, re.IGNORECASE)
                for location in locations
            ):
                continue

            entities.append((ent.text, label))

    for location in locations:
        if re.search(r"(?<!\w)" + re.escape(location) + r"(?!\w)", text, re.IGNORECASE):
            if not any(value.lower() == location.lower() and label == "LOC" for value, label in entities):
                entities.append((location, "LOC"))

    for organization in organizations:
        if re.search(r"(?<!\w)" + re.escape(organization) + r"(?!\w)", text, re.IGNORECASE):
            if not any(value.lower() == organization.lower() and label == "ORG" for value, label in entities):
                entities.append((organization, "ORG"))

    return entities

for ticket_id in ["IMH-004", "IMH-011", "IMH-018"]:
    row = df[df["ticket_id"] == ticket_id].iloc[0]
    print("\nTicket:", ticket_id)
    print("Complaint:", row["complaint_text"])
    print("Entities:", extract_entities(row["complaint_text"]))
print("\nGold entities from dataset:")

for ticket_id in ["IMH-004", "IMH-011", "IMH-018"]:
    row = df[df["ticket_id"] == ticket_id].iloc[0]
    print(ticket_id, ":", row["gold_entities"])

true_positive = 0
false_positive = 0
false_negative = 0

for _, row in df.iterrows():
    predicted = set(
        (value.strip().casefold(), label.strip().upper())
        for value, label in extract_entities(str(row["complaint_text"]))
    )

    gold_text = row["gold_entities"]

    gold = set()

    if pd.notna(gold_text):
        for item in str(gold_text).split(";"):
            parts = item.strip().rsplit("|", 1)

            if len(parts) == 2:
                entity, label = parts
                gold.add((entity.strip().casefold(), label.strip().upper()))

    true_positive += len(predicted & gold)
    false_positive += len(predicted - gold)
    false_negative += len(gold - predicted)

precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0
recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

print("\nNER Evaluation on All 40 Tickets")
print("Total tickets:", len(df))
print("True positives:", true_positive)
print("False positives:", false_positive)
print("False negatives:", false_negative)
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-score:", round(f1, 4))

print("\nNER Error Analysis")

for _, row in df.iterrows():
    predicted = set(
        (value.strip().casefold(), label.strip().upper())
        for value, label in extract_entities(str(row["complaint_text"]))
    )

    gold = set()
    gold_text = row["gold_entities"]

    if pd.notna(gold_text):
        for item in str(gold_text).split(";"):
            parts = item.strip().rsplit("|", 1)

            if len(parts) == 2:
                entity, label = parts
                gold.add((entity.strip().casefold(), label.strip().upper()))

    false_positives = predicted - gold
    false_negatives = gold - predicted

    if false_positives or false_negatives:
        print("\nTicket:", row["ticket_id"])
        print("Complaint:", row["complaint_text"])
        print("Gold:", sorted(gold))
        print("Predicted:", sorted(predicted))
        print("False positives:", sorted(false_positives))
        print("False negatives:", sorted(false_negatives))
        
print("\nNER Evaluation by Entity Type")

for entity_type in ["PER", "LOC", "ORG"]:
    tp = 0
    fp = 0
    fn = 0

    for _, row in df.iterrows():
        predicted = {
            (value.strip().casefold(), label.strip().upper())
            for value, label in extract_entities(str(row["complaint_text"]))
            if label.strip().upper() == entity_type
        }

        gold = set()
        gold_text = row["gold_entities"]

        if pd.notna(gold_text):
            for item in str(gold_text).split(";"):
                parts = item.strip().rsplit("|", 1)

                if len(parts) == 2:
                    entity, label = parts
                    if label.strip().upper() == entity_type:
                        gold.add((entity.strip().casefold(), entity_type))

        tp += len(predicted & gold)
        fp += len(predicted - gold)
        fn += len(gold - predicted)

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    print(f"\n{entity_type}")
    print("TP:", tp, "FP:", fp, "FN:", fn)
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))
