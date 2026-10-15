# Fake News Awareness & Detection System
Mini Project 2026-27 · Department of Computer Applications

## Run
```
pip install -r requirements.txt
python generate_dataset.py     # optional: rebuilds the demo dataset
python train_model.py          # trains model, fills SQLite, writes metrics
streamlit run app.py
```
(The app trains automatically on first launch if no model exists.)

## Structure
- `app.py` – Streamlit UI (Detector, Dashboard, Awareness Guide, Model Performance, History, About)
- `core/preprocess.py` – text cleaning, stop-word removal, red-flag rules
- `core/model.py` – TF-IDF + Logistic Regression, evaluation, prediction
- `core/database.py` – SQLite (news_records, predictions)
- `data/news_dataset.csv` – local dataset (synthetic demo)

## Important
The bundled dataset is **synthetic**, so accuracy is unrealistically high. For your final report, replace
`data/news_dataset.csv` with a real dataset (e.g. Kaggle "Fake and Real News", columns text,label,topic) and retrain.
