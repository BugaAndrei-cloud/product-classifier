import pickle
import re

# Funcție curățare titlu
def clean_title(title):
    title = title.lower()
    title = re.sub(r'[^a-z0-9\s]', '', title)
    return title

# Încărcare model
with open("models/final_model.pkl", "rb") as f:
    model, vectorizer = pickle.load(f)

# Predicție interactivă
while True:
    title = input("Introdu titlul produsului (sau 'exit'): ")
    if title.lower() == "exit":
        break
    cleaned = clean_title(title)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    print(f"Categoria prezisă: {pred}\n")
