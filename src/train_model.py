# src/train_model.py
import argparse
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

from text_cleaning import clean_text_series

def build_pipeline():
    # Curățare + TF-IDF + Logistic Regression într-un singur pipeline
    return Pipeline([
        ("clean", FunctionTransformer(clean_text_series, validate=False)),
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),       # unigrame + bigrame (de obicei crește calitatea)
            min_df=2,                 # ignoră termeni foarte rari
            max_df=0.95,              # ignoră termeni prea frecvenți (zgomot)
            strip_accents="unicode",  # normalizează diacritice
        )),
        ("clf", LogisticRegression(
            random_state=42,
            class_weight="balanced",
            max_iter=2000,
            n_jobs=None,
        ))
    ])

def plot_confusion_matrix(cm, labels, out_path):
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = range(len(labels))
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def main(args):
    # Load
    df = pd.read_csv(args.input_csv)

    # Keep essentials
    df = df.dropna(subset=["Product Title", "Category Label"]).copy()
    X = df["Product Title"]
    y = df["Category Label"]

    # Split (stratify păstrează proporțiile pe clase)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # Build + fit
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    # Eval
    y_pred = pipe.predict(X_test)
    report = classification_report(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)
    with open(args.metrics_out, "w", encoding="utf-8") as f:
        f.write("Distribuția claselor în y_train:\n")
        f.write(str(Counter(y_train)) + "\n\n")
        f.write("Distribuția claselor în y_test:\n")
        f.write(str(Counter(y_test)) + "\n\n")
        f.write(report + "\n")
        f.write(f"Precision (macro): {precision:.4f}\n")
        f.write(f"Recall (macro):    {recall:.4f}\n")
        f.write(f"F1 (macro):        {f1:.4f}\n")

    # Confusion matrix (etichetăm cu etichete ordonate după apariție în test)
    labels = sorted(list(set(y_test)))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    os.makedirs(os.path.dirname(args.cm_out), exist_ok=True)
    plot_confusion_matrix(cm, labels, args.cm_out)

    # Save model
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    joblib.dump(pipe, args.model_out)

    print("Model salvat la:", args.model_out)
    print("Raport salvat la:", args.metrics_out)
    print("Matrice de confuzie salvată la:", args.cm_out)
    print(f"Macro F1: {f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save product category classifier.")
    parser.add_argument("--input_csv", type=str, default="data/products.csv", help="Calea către products.csv")
    parser.add_argument("--model_out", type=str, default="models/model.pkl", help="Unde salvăm modelul .pkl")
    parser.add_argument("--metrics_out", type=str, default="reports/metrics.txt", help="Unde salvăm raportul")
    parser.add_argument("--cm_out", type=str, default="reports/confusion_matrix.png", help="Unde salvăm matricea de confuzie")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proporția setului de test")
    parser.add_argument("--seed", type=int, default=42, help="Random state")
    args = parser.parse_args()
    main(args)
