import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
import re

DATA_PATH = "esf_label_sample_5000.csv"
MODEL_PATH = "models/tfidf_sgd_model.pkl"

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main():
    print("Читаем размеченные данные из:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)

    # фильтруем только те строки, где CATEGORY уже проставлен
    df = df[df["CATEGORY"].astype(str).str.strip() != ""].copy()
    print("Строк с размеченной категорией:", df.shape[0])

    if df.shape[0] < 100:
        print("❌ Слишком мало размеченных данных (<100). Размечаем ещё немного и повторяем.")
        return

    df["DESCRIPTION_CLEAN"] = df["DESCRIPTION"].astype(str).map(clean_text)

    X = df["DESCRIPTION_CLEAN"]
    y = df["CATEGORY"]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2
        )),
        ("clf", SGDClassifier(
            loss="log_loss",
            max_iter=50,
            n_jobs=-1,
            random_state=42
        )),
    ])

    print("Обучаем модель...")
    pipeline.fit(X_train, y_train)

    print("Оцениваем на валидации...")
    y_pred = pipeline.predict(X_valid)
    print(classification_report(y_valid, y_pred, digits=4))

    # сохраняем модель
    import os
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print("✅ Модель сохранена в", MODEL_PATH)

if __name__ == "__main__":
    main()
