# 🛒 Product Category Classifier

Soluție ML pentru **clasificarea automată a produselor pe categorii** pe baza titlului acestora.  
Proiectul demonstrează cum putem folosi algoritmi de Machine Learning pentru a automatiza etichetarea produselor într-un magazin online.

---

## 🔍 Prezentare generală

Acest proiect abordează o problemă reală de **e-commerce**: mii de produse sunt adăugate zilnic, iar clasificarea manuală este lentă și predispusă la erori.  
Prin acest sistem:
- Titlurile produselor sunt procesate și vectorizate (TF-IDF);
- Sunt antrenate și comparate mai multe modele ML (Logistic Regression, Naive Bayes);
- Cel mai bun model este salvat și folosit pentru predicții rapide.

Beneficii:
- **Automatizare completă** a clasificării produselor;
- **Scalabilitate** pentru baze de date mari;
- **Extensibilitate** pentru a integra modele avansate sau API-uri.

---

## 📂 Structura proiectului

product-classifier/
├── data/
│   └── products.csv
├── notebooks/
│   ├── 01_exploration_and_cleaning.ipynb
│   └── 02_modeling_and_evaluation.ipynb
├── src/
│   ├── text_cleaning.py
│   └── utils.py
├── scripts/
│   ├── train_model.py
│   ├── predict_category.py
│   └── evaluate_model.py
├── models/
│   └── model.pkl
├── reports/
│   ├── metrics.txt
│   └── confusion_matrix.png
├── requirements.txt
└── README.md

---

## ⚙️ Instalare și configurare

1. Clonează proiectul:
```bash
git clone https://github.com/BugaAndrei-cloud/product-classifier
cd product-classifier
Creează și activează un mediu virtual:

bash
Copiază codul
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
Instalează dependențele:

bash
Copiază codul
pip install -r requirements.txt
🚀 Utilizare
1. Antrenează și salvează modelul
bash
Copiază codul
python src/train_model.py
Modelul antrenat va fi salvat în models/model.pkl.

2. Prezicere categorie pentru un titlu
bash
Copiază codul
python src/predict_category.py
Introdu titlul produsului, iar scriptul va returna categoria estimată.

📊 Rezultate
Acuratețea modelului: ~98% 
Matricea de confuzie și metricile detaliate sunt salvate în reports/.

🛠️ Tehnologii folosite
Python 3.8+

pandas, numpy – manipulare date

scikit-learn – vectorizare text și modele ML

matplotlib, seaborn – vizualizare metrici

pickle – salvare model

🔮 Direcții viitoare
Curățare text avansată: eliminare stopwords, lematizare, stemming

Testare cu modele avansate (XGBoost, LightGBM, Transformers)

Crearea unui API Flask/FastAPI pentru integrare cu platforme e-commerce

Interfață grafică pentru utilizatori non-tehnici

👥 Contribuție
Pull request-urile sunt binevenite!
Pentru schimbări majore, deschide mai întâi un issue pentru a discuta modificările propuse.