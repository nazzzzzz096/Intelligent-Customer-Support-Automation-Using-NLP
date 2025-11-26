# src/preprocessing.py

import pandas as pd
import re
from sklearn.model_selection import train_test_split


def clean_text(text: str) -> str:
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)

    # Remove numbers (optional)
    text = re.sub(r"\d+", " ", text)

    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)

    # Remove repeated whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()



def load_and_prepare_data(path: str):
    df = pd.read_csv(path)

    # Rename for convenience
    df = df.rename(columns={
        "Consumer complaint narrative": "text",
        "Issue": "target"
    })

    # Clean text
    df["text"] = df["text"].astype(str).apply(clean_text)

        # ---- REMOVE RARE CLASSES -----
    class_counts = df["target"].value_counts()
    valid_classes = class_counts[class_counts >= 5].index  # keep classes with >=5 samples
    df = df[df["target"].isin(valid_classes)].reset_index(drop=True)
    print("Classes with <5 samples:")
    print(class_counts[class_counts < 5])

    return df[["text", "target"]]


def split_data(df, test_size=0.2, random_state=42):
    X = df["text"].values
    y = df["target"].values

    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
