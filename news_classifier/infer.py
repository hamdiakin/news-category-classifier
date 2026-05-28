import json

import torch

from news_classifier.model import TextCNN


class Predictor:
    def __init__(self, checkpoint_path, vocab_path, cat_to_idx_path, cfg):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(vocab_path) as f:
            self.vocab = json.load(f)

        with open(cat_to_idx_path) as f:
            cat_to_idx = json.load(f)
        self.idx_to_cat = {v: k for k, v in cat_to_idx.items()}

        self.model = TextCNN(
            vocab_size=len(self.vocab),
            embed_dim=cfg.model.embed_dim,
            num_classes=len(cat_to_idx),
            num_filters=cfg.model.num_filters,
            kernel_sizes=tuple(cfg.model.kernel_sizes),
            dropout=0.0,
        )

        state = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        if "state_dict" in state:
            state = {k.replace("model.", ""): v for k, v in state["state_dict"].items()}
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        self.max_len = cfg.max_len

    def predict(self, headline, description=""):
        text = f"{headline} {description}".lower().split()
        token_ids = [self.vocab.get(t, 1) for t in text[: self.max_len]]
        token_ids = token_ids + [0] * (self.max_len - len(token_ids))

        tokens = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits = self.model(tokens)
            probs = torch.softmax(logits, dim=1)
            confidence, pred_idx = probs.max(dim=1)

        category = self.idx_to_cat[pred_idx.item()]
        return {"category": category, "confidence": round(confidence.item(), 4)}
