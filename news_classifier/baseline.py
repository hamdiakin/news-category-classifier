import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

from news_classifier.data import download_data, load_and_clean, split_data


def run_tfidf_baseline(cfg):
    raw_path = download_data(cfg.data_dir)
    df, cat_to_idx = load_and_clean(raw_path)
    train_df, val_df, test_df = split_data(df, cfg.data_dir)

    idx_to_cat = {v: k for k, v in cat_to_idx.items()}

    vectorizer = TfidfVectorizer(max_features=cfg.baseline.tfidf_max_features, ngram_range=(1, 2))
    x_train = vectorizer.fit_transform(train_df["text"])
    x_val = vectorizer.transform(val_df["text"])

    clf = LogisticRegression(
        max_iter=cfg.baseline.logreg_max_iter, C=cfg.baseline.logreg_C, solver="lbfgs"
    )
    clf.fit(x_train, train_df["label"])

    val_preds = clf.predict(x_val)
    val_probs = clf.predict_proba(x_val)

    results = {
        "accuracy": accuracy_score(val_df["label"], val_preds),
        "macro_f1": f1_score(val_df["label"], val_preds, average="macro"),
        "macro_precision": precision_score(val_df["label"], val_preds, average="macro"),
        "macro_recall": recall_score(val_df["label"], val_preds, average="macro"),
        "weighted_f1": f1_score(val_df["label"], val_preds, average="weighted"),
        "log_loss": log_loss(val_df["label"], val_probs),
    }

    print("\nTF-IDF + LogReg baseline results:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")

    target_names = [idx_to_cat[i] for i in range(len(cat_to_idx))]
    print("\nPer-class report:")
    print(classification_report(val_df["label"], val_preds, target_names=target_names))

    output_dir = Path(cfg.data_dir) / "baseline_results"
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "tfidf_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    cm = confusion_matrix(val_df["label"], val_preds)
    np.save(output_dir / "tfidf_confusion_matrix.npy", cm)

    return results
