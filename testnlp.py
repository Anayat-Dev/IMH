import nltk
import pandas
import sklearn
import spacy

from nltk.tokenize import word_tokenize

text = "Water is leaking in Islamabad!"

print("Tokens:", word_tokenize(text))

nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

for entity in doc.ents:
    print("Entity:", entity.text, "| Type:", entity.label_)

print("All libraries are working!")