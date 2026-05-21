from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

MODEL_PATH = MODELS_DIR / "sentiment_pipeline.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.json"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"

# IMDB: 0 = negative, 1 = positive
LABEL_NAMES = {0: "Olumsuz", 1: "Olumlu"}

# Training subset for faster local runs (set to None for full dataset)
TRAIN_SAMPLE_SIZE = 10_000
TEST_SAMPLE_SIZE = 2_000

RANDOM_STATE = 42
