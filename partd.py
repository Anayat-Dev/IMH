
import pandas as pd
import spacy
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("imh_complaints.csv")
nlp = spacy.load("en_core_web_sm")

ticket_ids = ["IMH-004", "IMH-011", "IMH-018", "IMH-027", "IMH-033"]
output_dir = Path("figures")
output_dir.mkdir(exist_ok=True)

def constituency_sketch(ticket_id, text):
    sketches = {
        "IMH-004": "(S (NP The residents (PP of (NP G-9))) (VP submitted (NP a complaint) (PP about (NP the transformer))))",
        "IMH-011": "(S (NP CDA water tanker) (VP did not reach (NP Street 12) (NP yesterday)))",
        "IMH-018": "(S (NP Someone) (VP dumped (NP medical waste) (PP behind (NP PIMS hostel)))",
        "IMH-027": "(S (NP Traffic police) (VP blocked (NP the I-8 underpass) (PP during (NP peak hours))))",
        "IMH-033": "(S (NP IESCO staff) (VP replaced (NP the meter) (S but (NP billing) (VP is (ADJP still wrong))))"
    }
    return sketches[ticket_id]

def draw_dependency(doc, ticket_id):
    tokens = list(doc)
    positions = {token.i: (i, 0) for i, token in enumerate(tokens)}

    fig, ax = plt.subplots(figsize=(max(10, len(tokens) * 0.8), 4))
    ax.axis("off")

    for token in tokens:
        x, y = positions[token.i]
        ax.text(x, y, token.text, ha="center", va="center", fontsize=9)
        if token.head.i != token.i:
            hx, hy = positions[token.head.i]
            ax.annotate(
                "",
                xy=(hx, hy + 0.12),
                xytext=(x, y + 0.12),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.25")
            )
            ax.text((x + hx) / 2, 0.35, token.dep_, ha="center", fontsize=7)

    ax.set_xlim(-0.5, len(tokens) - 0.5)
    ax.set_ylim(-0.2, 1.0)
    plt.tight_layout()
    plt.savefig(output_dir / f"{ticket_id}_dependency.png", dpi=200)
    plt.close(fig)

for ticket_id in ticket_ids:
    row = df[df["ticket_id"] == ticket_id].iloc[0]
    text = str(row["complaint_text"])
    doc = nlp(text)

    print("\n" + "=" * 70)
    print("Ticket:", ticket_id)
    print("Sentence:", text)
    print("Constituency sketch:")
    print(constituency_sketch(ticket_id, text))
    print("\nDependency parse:")
    print(f"{'TOKEN':<20}{'POS':<12}{'HEAD':<20}{'DEPENDENCY'}")

    for token in doc:
        print(f"{token.text:<20}{token.pos_:<12}{token.head.text:<20}{token.dep_}")

    subject = next((t for t in doc if t.dep_ in ["nsubj", "nsubjpass"]), None)
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    objects = [t for t in doc if t.dep_ in ["dobj", "obj", "attr", "oprd"]]
    locations = [t for t in doc if t.dep_ in ["obl", "prep", "pobj"]]

    subject_text = subject.text if subject else "Not identified"
    predicate_text = root.text if root else "Not identified"
    object_text = objects[0].text if objects else (
        locations[0].text if locations else "Not identified"
    )

    print("\nTriple:")
    print((subject_text, predicate_text, object_text))

    if ticket_id in ["IMH-004", "IMH-027"]:
        draw_dependency(doc, ticket_id)


doc = nlp(str(df[df["ticket_id"] == "IMH-004"].iloc[0]["complaint_text"]))

print("\nTokens for gold annotation:")
for token in doc:
    print(token.i, token.text, "Predicted head:", token.head.text, "Label:", token.dep_)

gold_heads = {
    0: 1,
    1: 4,
    2: 1,
    3: 2,
    4: 4,
    5: 6,
    6: 4,
    7: 6,
    8: 9,
    9: 7,
    10: 4
}

gold_labels = {
    0: "det",
    1: "nsubj",
    2: "prep",
    3: "pobj",
    4: "ROOT",
    5: "det",
    6: "dobj",
    7: "prep",
    8: "det",
    9: "pobj",
    10: "punct"
}

if len(doc) != len(gold_heads):
    print("\nToken count differs from the gold annotation.")
    print("Update gold_heads and gold_labels to match the token list above.")

else:
    correct_heads = 0
    correct_labels = 0

    for token in doc:
        gold_head_index = gold_heads[token.i]
        gold_label = gold_labels[token.i]

        if token.head.i == gold_head_index:
            correct_heads += 1

            if token.dep_ == gold_label:
                correct_labels += 1

    n = len(doc)
    uas = correct_heads / n
    las = correct_labels / n

    print("\nUAS/LAS Evaluation for IMH-004")
    print("Tokens evaluated:", n)
    print("Correct heads:", correct_heads)
    print("Correct heads and labels:", correct_labels)
    print("UAS:", round(uas, 4))
    print("LAS:", round(las, 4))
    print("Note: LAS should not exceed UAS.")
