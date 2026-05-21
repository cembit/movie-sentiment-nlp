# Movie Sentiment NLP

Classify IMDB movie reviews as **positive** or **negative** using a classic NLP pipeline (TF-IDF + Logistic Regression), with a live Streamlit demo.

> **TR (kısa):** IMDB yorumlarında olumlu/olumsuz duygu analizi — portföy projesi.

## Live demo

**[Open the app](https://movie-sentiment-nlp-katv9befxlmprsrcw4kbyu.streamlit.app)** — no install required; type an English review and click **Analiz et**.

> **TR:** Linke tıklayan herkes tarayıcıdan deneyebilir.

![Streamlit UI](assets/streamlit-demo.jpg)

## Problem

Given an English movie review, predict sentiment: **0 = negative**, **1 = positive**.

## Tech stack

| Area | Tools |
|------|--------|
| Data | [Hugging Face IMDB](https://huggingface.co/datasets/imdb) |
| Preprocessing | NLTK (HTML removal, stopwords, lemmatization) |
| Model | TF-IDF (1–2 grams) + Logistic Regression |
| Pipeline | scikit-learn `Pipeline` |
| Deployment | Streamlit Community Cloud |

## Results

Trained on 10,000 samples; evaluated on 2,000 test samples.

| Metric | Validation | Test |
|--------|------------|------|
| **Accuracy** | 87.1% | **87.1%** |
| **Precision** | 85.4% | 84.6% |
| **Recall** | 89.1% | 89.3% |
| **F1-score** | 0.87 | **0.87** |

Validation and test scores are close → weak sign of overfitting.

### Confusion matrix (test set)

![Confusion Matrix](reports/confusion_matrix.png)

| | Pred. negative | Pred. positive |
|--|----------------|----------------|
| **True negative** | 884 | 156 |
| **True positive** | 103 | 857 |

## Example predictions

```bash
python -m src.predict "This movie was terrible and boring."
# Negative (confidence: 97.74%)

python -m src.predict "Amazing film, best acting ever!"
# Positive (confidence: 77.96%)
```

## Project structure

```
movie-sentiment-nlp/
├── assets/
│   └── streamlit-demo.jpg
├── src/
│   ├── config.py
│   ├── load_data.py       # load IMDB from Hugging Face
│   ├── preprocess.py      # text cleaning
│   ├── train_model.py     # train + save metrics/plots
│   └── predict.py         # single-review inference
├── app.py                 # Streamlit UI
├── reports/
│   ├── metrics.json
│   └── confusion_matrix.png
├── models/
│   └── sentiment_pipeline.joblib
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/cembit/movie-sentiment-nlp.git
cd movie-sentiment-nlp
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

## Train (optional)

The trained model is included in the repo. To retrain:

```bash
python -m src.train_model
```

First run downloads IMDB (internet required). Outputs:

- `models/sentiment_pipeline.joblib`
- `reports/metrics.json`
- `reports/confusion_matrix.png`

Full dataset — set in `src/config.py`:

```python
TRAIN_SAMPLE_SIZE = None
TEST_SAMPLE_SIZE = None
```

## Run locally

```bash
streamlit run app.py
```

## What I learned

- End-to-end text classification with sklearn `Pipeline`
- TF-IDF features and bigrams
- Class imbalance handling (`class_weight='balanced'`)
- Train / validation / test split and F1 score
- Packaging a model with `joblib` and deploying on Streamlit Cloud

> **TR:** Klasik NLP + canlı demo; sonraki adım DistilBERT ile karşılaştırma.

## Roadmap

- Fine-tune **DistilBERT** on the same dataset
- Compare classical ML vs transformer metrics

## License

Educational / portfolio use. IMDB dataset via Hugging Face.
