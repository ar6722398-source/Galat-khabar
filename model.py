"""Training, evaluation and prediction: TF-IDF + Logistic Regression."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split

from .preprocess import clean_text
from . import database

BASE = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE / "models" / "model.joblib"
METRICS_PATH = BASE / "models" / "metrics.json"
DATA_PATH = BASE / "data" / "news_dataset.csv"


def train(data_path=DATA_PATH, test_size=0.2, seed=42):
    df = pd.read_csv(data_path).dropna(subset=["text", "label"])
    database.load_dataset_into_db(df)
    df["clean"] = df["text"].apply(clean_text)
    y = (df["label"].str.lower() == "fake").astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(df["clean"], y, test_size=test_size,
                                              random_state=seed, stratify=y)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xtr = vec.fit_transform(X_tr)
    clf = LogisticRegression(max_iter=1000, C=2.0)
    clf.fit(Xtr, y_tr)
    pred = clf.predict(vec.transform(X_te))
    p, r, f, _ = precision_recall_fscore_support(y_te, pred, average="binary", zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(y_te, pred)),
        "precision": float(p), "recall": float(r), "f1": float(f),
        "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
        "train_size": int(len(X_tr)), "test_size": int(len(X_te)),
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"vectorizer": vec, "classifier": clf}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return metrics


def load_model():
    if not MODEL_PATH.exists():
        train()
    return joblib.load(MODEL_PATH)


def load_metrics():
    if not METRICS_PATH.exists():
        train()
    return json.loads(METRICS_PATH.read_text())


def predict(text, bundle=None, top_n=6):
    """Return label, fake-probability and the words that pushed the decision."""
    bundle = bundle or load_model()
    vec, clf = bundle["vectorizer"], bundle["classifier"]
    X = vec.transform([clean_text(text)])
    p_fake = float(clf.predict_proba(X)[0, 1])
    is_fake = p_fake >= 0.5
    names = np.array(vec.get_feature_names_out())
    row = X.tocoo()
    sign = 1 if is_fake else -1
    contrib = {names[j]: v * clf.coef_[0][j] for j, v in zip(row.col, row.data)}
    top = sorted(contrib.items(), key=lambda kv: sign * kv[1], reverse=True)
    return {
        "label": "Potentially Fake" if is_fake else "Real",
        "p_fake": p_fake,
        "confidence": p_fake if is_fake else 1 - p_fake,
        "top_words": [w for w, v in top if sign * v > 0][:top_n],
        "known_terms": int(row.nnz),
    }
