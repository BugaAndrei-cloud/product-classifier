# src/text_cleaning.py
import re
import pandas as pd

def clean_text_series(s: pd.Series) -> pd.Series:
    """
    Curăță textul: litere mici, păstrează litere/cifre/spații, strip.
    Primește/returnează un pd.Series (potrivit pentru FunctionTransformer).
    """
    s = s.fillna("").astype(str).str.lower()
    s = s.apply(lambda x: re.sub(r"[^a-z0-9\s]+", " ", x))
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s
