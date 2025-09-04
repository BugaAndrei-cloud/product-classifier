import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import re

# Curățare titluri
def clean_title(title):
    title = title.lower()
    title = re.sub(r'[^a-z0-9\s]', '', title)
    return title

# Încărcăm datele
df = pd.read_csv("data/products.csv")
df.dropna(subset=["Product Title", "Category Label"], inplace=True)
df["Product Title"] = df["Product Title"].apply(clean_title)

# Split date
X = df["Product Title"]
y = df["Category Label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

# Model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_vec, y_train)

# Salvăm
with open("models/final_model.pkl", "wb") as f:
    pickle.dump((model, vectorizer), f)

print("Modelul a fost antrenat și salvat!")
