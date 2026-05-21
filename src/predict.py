"""Predict sentiment for a single review."""

import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import LABEL_NAMES, MODEL_PATH  # noqa: E402
from src.preprocess import clean_text  # noqa: E402


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model bulunamadı: {MODEL_PATH}\n"
            "Önce çalıştırın: python -m src.train_model"
        )
    return joblib.load(MODEL_PATH)


def predict_sentiment(text: str, pipeline=None) -> dict:
    if pipeline is None:
        pipeline = load_model()
    cleaned = clean_text(text)
    label = int(pipeline.predict([cleaned])[0])
    proba = pipeline.predict_proba([cleaned])[0]
    return {
        "label": label,
        "sentiment": LABEL_NAMES[label],
        "confidence": float(proba[label]),
        "prob_negative": float(proba[0]),
        "prob_positive": float(proba[1]),
    }


if __name__ == "__main__":
    sample = (
        "This movie was absolutely fantastic. Great acting and a gripping story."
        if len(sys.argv) < 2
        else " ".join(sys.argv[1:])
    )
    result = predict_sentiment(sample)
    print(f"Metin: {sample[:80]}...")
    print(f"Duygu: {result['sentiment']} (güven: {result['confidence']:.2%})")
