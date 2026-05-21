"""Load IMDB movie reviews from Hugging Face datasets."""

from typing import Tuple

import pandas as pd
from datasets import load_dataset

from src.config import RANDOM_STATE, TEST_SAMPLE_SIZE, TRAIN_SAMPLE_SIZE


def _to_dataframe(split, sample_size: int | None) -> pd.DataFrame:
    df = split.to_pandas()
    if sample_size is not None and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=RANDOM_STATE)
    return df.reset_index(drop=True)


def load_imdb(
    train_sample: int | None = TRAIN_SAMPLE_SIZE,
    test_sample: int | None = TEST_SAMPLE_SIZE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns train and test DataFrames with columns: text, label.
    label: 0 = negative, 1 = positive
    """
    dataset = load_dataset("imdb")
    train_df = _to_dataframe(dataset["train"], train_sample)
    test_df = _to_dataframe(dataset["test"], test_sample)
    return train_df, test_df
