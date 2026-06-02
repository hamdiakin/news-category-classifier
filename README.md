# News Category Classifier

Classifies news articles into one of 30 topic categories using the headline and a short description. The dataset has around 210k articles from Huffington Post (2012-2022).

I'm using a TextCNN with GloVe embeddings as the main model. There's also a TF-IDF + Logistic Regression baseline and a DistilBERT baseline for comparison.

## Setup

Python 3.10+ required.

```bash
git clone https://github.com/hamdiakin/news-category-classifier.git
cd news-category-classifier
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pre-commit install
```

## Data

You need a Kaggle API token at `~/.kaggle/kaggle.json`. The dataset downloads automatically on first training run, or you can grab it manually:

```bash
python commands.py download
```

If data is already tracked with DVC:

```bash
dvc pull
```

Download GloVe embeddings (optional but recommended):

```bash
curl -L -o data/glove.6B.zip https://nlp.stanford.edu/data/glove.6B.zip
unzip data/glove.6B.zip -d data/
rm data/glove.6B.zip data/glove.6B.50d.txt data/glove.6B.200d.txt data/glove.6B.300d.txt
```

## Train

Start MLflow first (in a separate terminal):

```bash
mlflow server --host 127.0.0.1 --port 8080
```

Then run training:

```bash
python commands.py train
```

Override config values if needed:

```bash
python commands.py train --overrides="[training.lr=0.0005, training.max_epochs=30]"
```

Run the TF-IDF baseline:

```bash
python commands.py baseline
```

After training, plots are saved to `plots/` and metrics are logged to MLflow (including hyperparams and git commit hash).

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
│   ├── training/default.yaml
│   └── serving/default.yaml
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
