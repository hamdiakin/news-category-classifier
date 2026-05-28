# News Category Classifier

Classifies news articles into one of 30 topic categories using the headline and a short description. The dataset has around 210k articles from Huffington Post (2012-2022).

I'm using a TextCNN with GloVe embeddings as the main model. There's also a TF-IDF + Logistic Regression baseline and a DistilBERT baseline for comparison.

## Setup

Python 3.10+ required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pre-commit install
```

If you want pretrained embeddings for TextCNN, grab GloVe:

```bash
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d data/
```

## Data

Dataset gets downloaded from Kaggle automatically the first time you run training. You'll need a Kaggle API token at `~/.kaggle/kaggle.json`.

You can also download it manually:

```bash
python commands.py download
```

Data and model files are tracked with DVC. Pull them with:

```bash
dvc pull
```

## Train

```bash
python commands.py train
```

You can override config values from the command line:

```bash
python commands.py train --overrides="[training.lr=0.0005, training.max_epochs=30]"
```

To run the TF-IDF baseline instead:

```bash
python commands.py baseline
```

Metrics get logged to MLflow. Start the server before training:

```bash
mlflow server --host 127.0.0.1 --port 8080
```

Training plots end up in `plots/`.

## Inference

```bash
python commands.py infer --headline="Democrats push new climate legislation" --description="Senate votes on clean energy bill"
```

## Project Structure

```
news-category-classifier/
├── commands.py
├── configs/
│   ├── defaults.yaml
│   ├── model/textcnn.yaml
│   └── training/default.yaml
├── news_classifier/
│   ├── data.py
│   ├── model.py
│   ├── trainer.py
│   ├── train.py
│   ├── baseline.py
│   ├── infer.py
│   └── plotting.py
├── .dvc/config
├── .pre-commit-config.yaml
└── pyproject.toml
```

## Metrics

Primary metric is Macro-F1 since the dataset has heavy class imbalance. I also track Accuracy, Macro-Precision, Macro-Recall, Weighted-F1, per-class F1 scores, confusion matrix, and log-loss.
