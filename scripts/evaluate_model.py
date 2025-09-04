# evaluate_model.py
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 1. Încărcăm modelul și vectorizatorul ---
with open("models/final_model.pkl", "rb") as f:
    final_model, vectorizer = pickle.load(f)

# --- 2. Încărcăm datele ---
df = pd.read_csv("data/products.csv")

# --- 3. Definim X și y și curățăm NaN ---
X = df["Product Title"].fillna("")      # Înlocuim NaN cu string gol
y = df["Category Label"].fillna("Unknown")  # Înlocuim NaN cu "Unknown"

# Împărțim datele în train/test (nu vom folosi train aici, dar păstrăm structura)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. Vectorizare set test ---
X_test_vectorized = vectorizer.transform(X_test)

# --- 5. Predicții și evaluare ---
y_pred = final_model.predict(X_test_vectorized)

# Raport clasificare
report = classification_report(y_test, y_pred, zero_division=0)
print(report)

# --- 6. Matrice de confuzie ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12,8))
sns.heatmap(cm, annot=False, cmap="Blues")
plt.title("Matrice de confuzie")
plt.xlabel("Predicted")
plt.ylabel("Actual")

# Salvează matricea de confuzie
os.makedirs("reports", exist_ok=True)
plt.savefig("reports/confusion_matrix.png")
plt.close()

# --- 7. Salvează raportul ---
with open("reports/metrics.txt", "w", encoding="utf-8") as f:
    f.write(report)

print("Evaluare completă! Rezultatele au fost salvate în folderul 'reports/'.")
