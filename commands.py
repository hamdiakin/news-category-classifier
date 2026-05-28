from pathlib import Path

import fire
from hydra import compose, initialize_config_dir


def _load_cfg(overrides=None):
    config_dir = str(Path(__file__).parent / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="defaults", overrides=overrides or [])
    return cfg


def train(overrides=None):
    cfg = _load_cfg(overrides)
    from news_classifier.train import train as run_train

    run_train(cfg)


def baseline(overrides=None):
    cfg = _load_cfg(overrides)
    from news_classifier.baseline import run_tfidf_baseline

    run_tfidf_baseline(cfg)


def download(data_dir="data"):
    from news_classifier.data import download_data

    path = download_data(data_dir)
    print(f"Data downloaded to {path}")


def infer(headline, description="", overrides=None):
    cfg = _load_cfg(overrides)
    from news_classifier.infer import Predictor

    predictor = Predictor(
        checkpoint_path=cfg.serving.checkpoint,
        vocab_path=cfg.serving.vocab_path,
        cat_to_idx_path=cfg.serving.cat_to_idx_path,
        cfg=cfg,
    )
    result = predictor.predict(headline, description)
    print(result)


if __name__ == "__main__":
    fire.Fire(
        {
            "train": train,
            "baseline": baseline,
            "download": download,
            "infer": infer,
        }
    )
