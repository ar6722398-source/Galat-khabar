"""Train the TF-IDF + Logistic Regression model:  python train_model.py"""
from core.model import train

if __name__ == "__main__":
    m = train()
    print(f"Accuracy {m['accuracy']:.3f} | Precision {m['precision']:.3f} | Recall {m['recall']:.3f} | F1 {m['f1']:.3f}")
    print("Confusion matrix [[TN, FP],[FN, TP]]:", m["confusion_matrix"])
