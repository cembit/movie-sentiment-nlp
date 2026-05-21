"""Streamlit UI for movie review sentiment prediction."""

import json
from pathlib import Path

import streamlit as st

from src.config import CONFUSION_MATRIX_PATH, METRICS_PATH
from src.predict import load_model, predict_sentiment

st.set_page_config(page_title="Film Yorumu Duygu Analizi", page_icon="🎬", layout="centered")

st.title("🎬 Film Yorumu Duygu Analizi")
st.caption("IMDB yorumları ile eğitilmiş TF-IDF + Logistic Regression modeli")

try:
    pipeline = load_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

review = st.text_area(
    "İngilizce film yorumu yazın",
    height=150,
    placeholder="This movie was boring and predictable...",
)

if st.button("Analiz et", type="primary") and review.strip():
    result = predict_sentiment(review, pipeline=pipeline)
    if result["label"] == 1:
        st.success(f"**{result['sentiment']}** — güven: {result['confidence']:.1%}")
    else:
        st.error(f"**{result['sentiment']}** — güven: {result['confidence']:.1%}")

    col1, col2 = st.columns(2)
    col1.metric("Olumsuz olasılık", f"{result['prob_negative']:.1%}")
    col2.metric("Olumlu olasılık", f"{result['prob_positive']:.1%}")

with st.expander("Model metrikleri"):
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        st.json(metrics)
    else:
        st.info("Henüz eğitim yapılmamış. `python -m src.train_model` çalıştırın.")

    if CONFUSION_MATRIX_PATH.exists():
        st.image(str(CONFUSION_MATRIX_PATH), caption="Test seti confusion matrix")
