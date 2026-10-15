"""Fake News Awareness & Detection System - Streamlit app.
Run:  streamlit run app.py
"""
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from core import database
from core.model import load_model, load_metrics, train, predict
from core.preprocess import red_flags

st.set_page_config(page_title="Fake News Awareness System", page_icon="📰", layout="wide")
database.init_db()


@st.cache_resource
def get_bundle():
    return load_model()


SAMPLES = {
    "— choose a sample —": "",
    "Sample 1 (sensational)": "SHOCKING!!! Doctors don't want you to know this miracle cure the government is hiding. Share before it is deleted!!!",
    "Sample 2 (reported news)": "The health ministry said on Tuesday that a new dengue prevention programme was approved, according to an official statement released in the capital.",
}

page = st.sidebar.radio("Navigate", ["🔍 News Detector", "📊 Awareness Dashboard", "🛡️ Awareness Guide",
                                     "📈 Model Performance", "🗂️ Prediction History", "ℹ️ About"])
st.sidebar.caption("Mini Project 2026-27 · Dept. of Computer Applications")

# ---------------------------------------------------------------- Detector
if page == "🔍 News Detector":
    st.title("📰 Fake News Awareness & Detection")
    st.write("Paste a headline or news paragraph. The model gives a **Real / Potentially Fake** prediction "
             "and explains the warning signs. Always verify important news with trusted sources.")
    choice = st.selectbox("Try a sample", list(SAMPLES))
    text = st.text_area("News text", value=SAMPLES[choice], height=170, placeholder="Paste news content here...")
    if st.button("Analyse", type="primary"):
        if len(text.strip()) < 15:
            st.warning("Please enter at least a full sentence.")
        else:
            res = predict(text, get_bundle())
            database.save_prediction(text, res["label"], res["confidence"])
            c1, c2, c3 = st.columns(3)
            c1.metric("Prediction", res["label"])
            c2.metric("Confidence", f"{res['confidence']*100:.1f}%")
            c3.metric("Fake probability", f"{res['p_fake']*100:.1f}%")
            st.progress(min(max(res["p_fake"], 0.0), 1.0), text="Fake-probability meter")
            if res["label"] == "Real":
                st.success("The text looks like regular reporting. This is NOT proof it is true — check the source.")
            else:
                st.error("This text shows patterns common in misinformation. Verify before sharing.")
            if res["known_terms"] == 0:
                st.info("None of the words were in the model's vocabulary, so this prediction is unreliable.")
            if res["top_words"]:
                st.write("**Words that influenced the result:** " + ", ".join(f"`{w}`" for w in res["top_words"]))
            flags = red_flags(text)
            st.subheader("Warning-sign check")
            if flags:
                for f in flags:
                    st.write("⚠️ " + f)
            else:
                st.write("✅ No obvious warning signs found.")
            st.caption("This tool assists human judgement; it can be wrong, especially on topics outside its training data.")

# ---------------------------------------------------------------- Dashboard
elif page == "📊 Awareness Dashboard":
    st.title("📊 Fake News Awareness Dashboard")
    rec = database.get_news_records()
    hist = database.get_predictions()
    a, b, c = st.columns(3)
    a.metric("Training news records", len(rec))
    b.metric("Checks performed", len(hist))
    c.metric("Flagged as fake", int((hist["prediction"] == "Potentially Fake").sum()) if len(hist) else 0)
    left, right = st.columns(2)
    if len(rec):
        with left:
            st.subheader("Dataset: Real vs Fake")
            fig, ax = plt.subplots(figsize=(4.5, 3.2))
            counts = rec["label"].value_counts()
            ax.bar(counts.index, counts.values, color=["#2a9d8f" if i == "Real" else "#d62839" for i in counts.index])
            ax.set_ylabel("Articles")
            st.pyplot(fig)
        with right:
            st.subheader("Fake share by topic")
            share = rec.assign(fake=(rec["label"] == "Fake")).groupby("topic")["fake"].mean() * 100
            fig, ax = plt.subplots(figsize=(4.5, 3.2))
            ax.barh(share.index, share.values, color="#f4a261")
            ax.set_xlabel("% fake")
            st.pyplot(fig)
    if len(hist):
        st.subheader("Your checks")
        fig, ax = plt.subplots(figsize=(4.5, 3))
        hist["prediction"].value_counts().plot.pie(autopct="%1.0f%%", ax=ax, colors=["#d62839", "#2a9d8f"], ylabel="")
        st.pyplot(fig)
    else:
        st.info("Run a few checks in the News Detector to see your own statistics here.")

# ---------------------------------------------------------------- Guide
elif page == "🛡️ Awareness Guide":
    st.title("🛡️ How to Spot Fake News")
    st.subheader("Verification checklist")
    for i, item in enumerate([
        "**Check the source** – Is it a known outlet? Look for an 'About' page and contact details.",
        "**Read beyond the headline** – Headlines are often exaggerated; the body may say something different.",
        "**Look for the author and date** – Old stories are often re-shared as if they were new.",
        "**Cross-check** – Do at least two reliable outlets report the same thing?",
        "**Check images and videos** – Use a reverse image search; images may be old or edited.",
        "**Beware of emotion** – Anger, fear and urgency ('share now!') are manipulation tactics.",
        "**Use fact-checkers** – e.g. PIB Fact Check, Alt News, BOOM, Snopes, AFP Fact Check.",
        "**Pause before sharing** – If you cannot verify it, do not forward it."], 1):
        st.write(f"{i}. {item}")
    st.subheader("Common types of misinformation")
    for t, d in {"Clickbait": "Sensational headline designed to get clicks.",
                 "Fabricated content": "Completely false content made to deceive.",
                 "Misleading content": "Real facts or quotes used out of context.",
                 "Manipulated media": "Edited photos, videos or deepfakes.",
                 "Satire taken seriously": "Humour shared as if it were real news.",
                 "Imposter content": "Fake pages imitating real news brands or officials."}.items():
        with st.expander(t):
            st.write(d)
    st.subheader("Quick quiz")
    q = st.radio("A message says 'FORWARD THIS to 10 people or your account will be blocked!!!'. What should you do?",
                 ["Forward it quickly to be safe", "Ignore and verify on the official website", "Reply asking for proof"], index=None)
    if q:
        st.success("Correct! Urgent threats are classic hoax tactics.") if q.startswith("Ignore") else st.error("Not quite — verify on the official source instead.")

# ---------------------------------------------------------------- Performance
elif page == "📈 Model Performance":
    st.title("📈 Model Testing & Evaluation")
    m = load_metrics()
    cols = st.columns(4)
    for col, k in zip(cols, ["accuracy", "precision", "recall", "f1"]):
        col.metric(k.upper() if k == "f1" else k.capitalize(), f"{m[k]*100:.1f}%")
    st.caption(f"Trained on {m['train_size']} records, tested on {m['test_size']} unseen records.")
    cm = m["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Real", "Fake"]); ax.set_yticks([0, 1], ["Real", "Fake"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i][j], ha="center", va="center", color="black")
    st.pyplot(fig)
    st.warning("Scores on the bundled demo dataset are very high because it is small and stylistically clear. "
               "Expect lower accuracy on real-world news — retrain with a larger dataset for the final evaluation.")
    st.subheader("Retrain model")
    up = st.file_uploader("Upload a CSV with columns: text, label (Real/Fake), topic", type="csv")
    if up is not None and st.button("Retrain with this dataset"):
        df = pd.read_csv(up)
        if not {"text", "label"} <= set(df.columns):
            st.error("CSV must contain 'text' and 'label' columns.")
        else:
            if "topic" not in df.columns:
                df["topic"] = "General"
            df.to_csv("data/news_dataset.csv", index=False)
            train()
            st.cache_resource.clear()
            st.success("Model retrained. Reload this page to see new metrics.")

# ---------------------------------------------------------------- History
elif page == "🗂️ Prediction History":
    st.title("🗂️ Prediction History (SQLite)")
    hist = database.get_predictions()
    if len(hist):
        hist["confidence"] = (hist["confidence"] * 100).round(1).astype(str) + "%"
        st.dataframe(hist, use_container_width=True, hide_index=True)
        st.download_button("Download CSV", hist.to_csv(index=False), "predictions.csv", "text/csv")
        if st.button("Clear history"):
            database.clear_predictions()
            st.rerun()
    else:
        st.info("No predictions saved yet.")

# ---------------------------------------------------------------- About
else:
    st.title("ℹ️ About the Project")
    st.write("**Fake News Awareness & Detection System** — Mini Project 2026-27, Department of Computer Applications "
             "(Domain: Data Science · Category: Application Development).")
    st.markdown("""
**Pipeline:** Data Collection → Preprocessing → NLP → TF-IDF → Model Training → Testing → Prediction → Result Display → Awareness Resources → SQLite Storage

**Stack:** Python, Streamlit, Pandas, NumPy, Scikit-learn (TF-IDF + Logistic Regression), SQLite, Matplotlib

**Team:** Ayush (Team Leader – ML & data processing) · Ayush Rana (Web development & database)  
**Preferred mentor:** Mr. Gaurav Sharma
""")
