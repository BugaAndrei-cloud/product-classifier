# src/predict_category.py
import argparse
import joblib
import sys

# IMPORTANT: pipeline-ul pickled conține referință la clean_text_series
# importat de în timpul antrenării. Deoarece funcția e definită în
# `text_cleaning.py` și importată în `train_model.py`, e suficient să
# existe în sys.modules când încărcăm modelul. De aceea importăm aici:
import text_cleaning  # noqa: F401  (nu folosit direct, dar necesar la unpickling)

def predict_interactive(model_path: str):
    pipe = joblib.load(model_path)
    print("Model încărcat. Introdu titluri (ENTER gol pentru a ieși).")
    while True:
        try:
            title = input("Titlu produs: ").strip()
        except EOFError:
            break
        if not title:
            break
        pred = pipe.predict([title])[0]
        proba = None
        # Dacă vrem și probabilități:
        if hasattr(pipe.named_steps["clf"], "predict_proba"):
            # vectorizăm prin pipeline automat
            probs = pipe.predict_proba([title])[0]
            # top-3 sugestii
            top3_idx = probs.argsort()[-3:][::-1]
            classes = pipe.named_steps["clf"].classes_
            top3 = [(classes[i], float(probs[i])) for i in top3_idx]
            print("Categorie prezisă:", pred)
            print("Top-3 sugestii:", top3)
        else:
            print("Categorie prezisă:", pred)

def predict_single(model_path: str, title: str):
    pipe = joblib.load(model_path)
    pred = pipe.predict([title])[0]
    out = {"title": title, "predicted_category": pred}
    # Optional: top-3
    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        probs = pipe.predict_proba([title])[0]
        top3_idx = probs.argsort()[-3:][::-1]
        classes = pipe.named_steps["clf"].classes_
        out["top3"] = [(classes[i], float(probs[i])) for i in top3_idx]
    print(out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict product category from title.")
    parser.add_argument("--model", type=str, default="models/model.pkl", help="Calea către modelul .pkl")
    parser.add_argument("--title", type=str, default=None, help="Titlul produsului (dacă e setat, rulează o singură predicție)")
    args = parser.parse_args()

    if args.title:
        predict_single(args.model, args.title)
    else:
        predict_interactive(args.model)
