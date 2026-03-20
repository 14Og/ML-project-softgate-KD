import re

import kagglehub
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch


def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()


class TextDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_imdb(
    max_features: int = 10_000,
    ngram_range: tuple = (1, 2),
    test_size: float = 0.2,
    batch_size: int = 256,
    random_state: int = 42,
) -> tuple[DataLoader, DataLoader, TfidfVectorizer, int]:
    """Download, preprocess and return train/test DataLoaders + fitted vectorizer + num_classes."""
    path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
    df = pd.read_csv(f"{path}/IMDB Dataset.csv")

    df["clean"] = df["review"].apply(clean)
    df["label"] = (df["sentiment"] == "positive").astype(int)

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["clean"].values,
        df["label"].values,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X_train = vectorizer.fit_transform(X_train_text).toarray().astype(np.float32)
    X_test = vectorizer.transform(X_test_text).toarray().astype(np.float32)

    train_loader = DataLoader(TextDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TextDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, vectorizer, 2


def load_20newsgroups(
    max_features: int = 10_000,
    ngram_range: tuple = (1, 2),
    batch_size: int = 256,
    random_state: int = 42,
) -> tuple[DataLoader, DataLoader, TfidfVectorizer, int]:
    """Load 20 Newsgroups, return train/test DataLoaders + fitted vectorizer + num_classes."""
    # remove=("headers", "footers", "quotes") strips email metadata that would
    # leak the newsgroup identity without the model reading actual content
    train_raw = fetch_20newsgroups(
        subset="train", remove=("headers", "footers", "quotes"), random_state=random_state,
    )
    test_raw = fetch_20newsgroups(
        subset="test", remove=("headers", "footers", "quotes"), random_state=random_state,
    )

    train_texts = [clean(t) for t in train_raw.data]
    test_texts = [clean(t) for t in test_raw.data]

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X_train = vectorizer.fit_transform(train_texts).toarray().astype(np.float32)
    X_test = vectorizer.transform(test_texts).toarray().astype(np.float32)

    y_train = train_raw.target.astype(np.int64)
    y_test = test_raw.target.astype(np.int64)

    train_loader = DataLoader(TextDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TextDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, vectorizer, 20
