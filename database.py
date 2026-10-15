"""SQLite storage for news records (training data) and prediction results."""
import sqlite3
from pathlib import Path
import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "fake_news.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS news_records(
            id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT, label TEXT, topic TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS predictions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            text TEXT, prediction TEXT, confidence REAL)""")


def load_dataset_into_db(df):
    """Replace stored news records with the dataset used for training."""
    init_db()
    cols = [c for c in ["text", "label", "topic"] if c in df.columns]
    with get_conn() as c:
        c.execute("DELETE FROM news_records")
        df[cols].to_sql("news_records", c, if_exists="append", index=False)


def save_prediction(text, prediction, confidence):
    init_db()
    with get_conn() as c:
        c.execute("INSERT INTO predictions(text, prediction, confidence) VALUES (?,?,?)",
                  (text, prediction, float(confidence)))


def get_predictions(limit=500):
    init_db()
    with get_conn() as c:
        return pd.read_sql_query(
            "SELECT id, created_at, text, prediction, confidence FROM predictions ORDER BY id DESC LIMIT ?",
            c, params=(limit,))


def clear_predictions():
    init_db()
    with get_conn() as c:
        c.execute("DELETE FROM predictions")


def get_news_records():
    init_db()
    with get_conn() as c:
        return pd.read_sql_query("SELECT * FROM news_records", c)
