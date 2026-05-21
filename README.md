# Movie Sentiment NLP

Classify IMDB movie reviews as **positive** or **negative** using TF-IDF + Logistic Regression. Includes a Streamlit web demo.

> **TR:** IMDB film yorumlarında olumlu / olumsuz sınıflandırma.

## Live demo

**[Open the app](https://movie-sentiment-nlp-katv9befxlmprsrcw4kbyu.streamlit.app)** — English review in, sentiment out. No local setup.

> **TR:** Linkten tarayıcıda denenebilir; `Analiz et` ile tahmin alınır.

![Streamlit UI](assets/streamlit-demo.jpg)

## Problem

Predict sentiment from an English movie review: **0 = negative**, **1 = positive**.

## Tech stack

| Area | Tools |
|------|--------|
| Data | [Hugging Face IMDB](https://huggingface.co/datasets/imdb) |
| Preprocessing | NLTK |
| Model | TF-IDF (1–2 grams) + Logistic Regression |
| Pipeline | scikit-learn |
| App | Streamlit |

## Results

10,000 training samples, 2,000 test samples.

| Metric | Validation | Test |
|--------|------------|------|
| Accuracy | 87.1% | 87.1% |
| Precision | 85.4% | 84.6% |
| Recall | 89.1% | 89.3% |
| F1 | 0.87 | 0.87 |

> **TR:** Doğrulama ve test skorları birbirine yakın.

### Confusion matrix (test)

![Confusion Matrix](reports/confusion_matrix.png)

| | Pred. negative | Pred. positive |
|--|----------------|----------------|
| True negative | 884 | 156 |
| True positive | 103 | 857 |

## Example predictions

```bash
python -m src.predict "This movie was terrible and boring."
# Negative — 97.74%

python -m src.predict "Amazing film, best acting ever!"
# Positive — 77.96%
```

## Project structure

```
movie-sentiment-nlp/
├── assets/streamlit-demo.jpg
├── src/
│   ├── config.py
│   ├── load_data.py
│   ├── preprocess.py
│   ├── train_model.py
│   └── predict.py
├── app.py
├── models/sentiment_pipeline.joblib
├── reports/
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/cembit/movie-sentiment-nlp.git
cd movie-sentiment-nlp
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

> **TR:** Windows’ta `venv\Scripts\activate`; macOS/Linux’ta `source venv/bin/activate`.

## Retrain

```bash
python -m src.train_model
```

Writes `models/sentiment_pipeline.joblib`, `reports/metrics.json`, `reports/confusion_matrix.png`. First run downloads IMDB.

Full dataset in `src/config.py`: set `TRAIN_SAMPLE_SIZE` and `TEST_SAMPLE_SIZE` to `None`.

## Run locally

```bash
.\venv\Scripts\activate
python -m streamlit run app.py
```

## License

IMDB data via [Hugging Face](https://huggingface.co/datasets/imdb).
