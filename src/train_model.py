"""Train TF-IDF + Logistic Regression sentiment classifier."""

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    CONFUSION_MATRIX_PATH,
    METRICS_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
)
from src.load_data import load_imdb  # noqa: E402
from src.preprocess import clean_text  # noqa: E402


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=20_000,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def save_confusion_matrix(y_true, y_pred, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Olumsuz", "Olumlu"],
        yticklabels=["Olumsuz", "Olumlu"],
        ax=ax,
    )
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    ax.set_title("Confusion Matrix — IMDB Sentiment")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    print("Veri seti yükleniyor (IMDB)...")
    train_df, test_df = load_imdb()

    print(f"Eğitim örnekleri: {len(train_df)} | Test: {len(test_df)}")
    print("Metinler temizleniyor (bu adım birkaç dakika sürebilir)...")

    train_df["text_clean"] = train_df["text"].apply(clean_text)
    test_df["text_clean"] = test_df["text"].apply(clean_text)

    X_train, X_val, y_train, y_val = train_test_split(
        train_df["text_clean"],
        train_df["label"],
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=train_df["label"],
    )

    pipeline = build_pipeline()
    print("Model eğitiliyor...")
    pipeline.fit(X_train, y_train)

    y_val_pred = pipeline.predict(X_val)
    y_test_pred = pipeline.predict(test_df["text_clean"])

    metrics = {
        "validation": {
            "accuracy": float(accuracy_score(y_val, y_val_pred)),
            "precision": float(precision_score(y_val, y_val_pred)),
            "recall": float(recall_score(y_val, y_val_pred)),
            "f1": float(f1_score(y_val, y_val_pred)),
        },
        "test": {
            "accuracy": float(accuracy_score(test_df["label"], y_test_pred)),
            "precision": float(precision_score(test_df["label"], y_test_pred)),
            "recall": float(recall_score(test_df["label"], y_test_pred)),
            "f1": float(f1_score(test_df["label"], y_test_pred)),
        },
        "train_samples": len(train_df),
        "test_samples": len(test_df),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    save_confusion_matrix(test_df["label"], y_test_pred, CONFUSION_MATRIX_PATH)

    print("\n--- Doğrulama (validation) ---")
    print(classification_report(y_val, y_val_pred, target_names=["Olumsuz", "Olumlu"]))

    print("--- Test ---")
    print(
        classification_report(
            test_df["label"], y_test_pred, target_names=["Olumsuz", "Olumlu"]
        )
    )

    print(f"\nModel kaydedildi: {MODEL_PATH}")
    print(f"Metrikler: {METRICS_PATH}")
    print(f"Confusion matrix: {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()
