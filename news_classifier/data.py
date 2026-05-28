import json
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

CATEGORY_MAP = {
    "ARTS": "ARTS & CULTURE",
    "CULTURE & ARTS": "ARTS & CULTURE",
    "PARENTS": "PARENTING",
    "STYLE": "STYLE & BEAUTY",
    "GREEN": "ENVIRONMENT",
    "TASTE": "FOOD & DRINK",
    "HEALTHY LIVING": "WELLNESS",
    "THE WORLDPOST": "WORLD NEWS",
    "WORLDPOST": "WORLD NEWS",
    "COLLEGE": "EDUCATION",
    "LATINO VOICES": "VOICES",
    "QUEER VOICES": "VOICES",
    "BLACK VOICES": "VOICES",
}


def _try_dvc_pull():
    import subprocess

    try:
        subprocess.run(["dvc", "pull"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def download_data(data_dir="data"):
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    raw_file = data_path / "news.json"
    if raw_file.exists():
        return raw_file

    _try_dvc_pull()
    if raw_file.exists():
        return raw_file

    dataset_path = kagglehub.dataset_download("rmisra/news-category-dataset")
    src = Path(dataset_path) / "News_Category_Dataset_v3.json"
    raw_file.write_bytes(src.read_bytes())
    return raw_file


def load_and_clean(raw_path):
    records = []
    with open(raw_path) as f:
        for line in f:
            records.append(json.loads(line))

    df = pd.DataFrame(records)
    df["category"] = df["category"].replace(CATEGORY_MAP)
    df["text"] = df["headline"] + " " + df["short_description"]
    df = df[["text", "category"]].dropna()

    categories = sorted(df["category"].unique())
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    df["label"] = df["category"].map(cat_to_idx)

    return df, cat_to_idx


def split_data(df, data_dir="data", val_ratio=0.1, test_ratio=0.1):
    split_file = Path(data_dir) / "split_indices.json"

    if split_file.exists():
        with open(split_file) as f:
            indices = json.load(f)
        train_idx = indices["train"]
        val_idx = indices["val"]
        test_idx = indices["test"]
    else:
        train_val_idx, test_idx = train_test_split(
            np.arange(len(df)),
            test_size=test_ratio,
            stratify=df["label"],
            random_state=42,
        )
        relative_val = val_ratio / (1 - test_ratio)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=relative_val,
            stratify=df["label"].iloc[train_val_idx],
            random_state=42,
        )
        train_idx = train_idx.tolist()
        val_idx = val_idx.tolist()
        test_idx = test_idx.tolist()

        with open(split_file, "w") as f:
            json.dump({"train": train_idx, "val": val_idx, "test": test_idx}, f)

    return df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]


class NewsDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=64):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.texts.iloc[idx].lower().split()
        token_ids = [self.vocab.get(t, 0) for t in tokens[: self.max_len]]

        padding = self.max_len - len(token_ids)
        token_ids = token_ids + [0] * padding

        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(
            self.labels.iloc[idx], dtype=torch.long
        )


def build_vocab(texts, min_freq=2):
    word_counts = {}
    for text in texts:
        for word in text.lower().split():
            word_counts[word] = word_counts.get(word, 0) + 1

    vocab = {"<pad>": 0, "<unk>": 1}
    idx = 2
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab[word] = idx
            idx += 1

    return vocab


def load_glove(glove_path, vocab, embed_dim=100):
    glove_path = Path(glove_path)
    embeddings = np.random.normal(0, 0.1, (len(vocab), embed_dim))
    embeddings[0] = 0  # pad

    found = 0
    with open(glove_path) as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            if word in vocab:
                embeddings[vocab[word]] = np.array(parts[1:], dtype=np.float32)
                found += 1

    print(f"Loaded {found}/{len(vocab)} words from GloVe")
    return torch.tensor(embeddings, dtype=torch.float32)


def get_dataloaders(cfg):
    raw_path = download_data(cfg.data_dir)
    df, cat_to_idx = load_and_clean(raw_path)
    train_df, val_df, test_df = split_data(df, cfg.data_dir)

    vocab = build_vocab(train_df["text"], min_freq=cfg.min_freq)

    train_set = NewsDataset(train_df["text"], train_df["label"], vocab, cfg.max_len)
    val_set = NewsDataset(val_df["text"], val_df["label"], vocab, cfg.max_len)
    test_set = NewsDataset(test_df["text"], test_df["label"], vocab, cfg.max_len)

    bs = cfg.training.batch_size
    nw = cfg.training.num_workers
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=nw)
    val_loader = DataLoader(val_set, batch_size=bs, num_workers=nw)
    test_loader = DataLoader(test_set, batch_size=bs, num_workers=nw)

    return train_loader, val_loader, test_loader, vocab, cat_to_idx
