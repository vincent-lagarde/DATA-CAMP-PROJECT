import torch
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import numpy as np


# Custom Dataset for Wine Reviews
class WineReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        # Convert texts and labels to lists to avoid indexing issues.
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# Custom scikit-learn Estimator wrapping BERT
class BertSklearnClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, pretrained_model_name="bert-base-uncased",
                 max_length=64, epochs=1, batch_size=16, learning_rate=3e-5,
                 device=None):
        self.pretrained_model_name = pretrained_model_name
        self.max_length = max_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device if device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = BertTokenizer.from_pretrained(
            self.pretrained_model_name
        )
        self.model = None  # Model will be created in fit()

    def fit(self, X, y):
        num_labels = len(np.unique(y))
        self.model = BertForSequenceClassification.from_pretrained(
            self.pretrained_model_name, num_labels=num_labels
        )
        self.model.to(self.device)
        self.model.train()

        dataset = WineReviewDataset(X, y, self.tokenizer,
                                    max_length=self.max_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size,
                                shuffle=True)

        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.epochs):
            total_loss = 0
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        return self

    def predict(self, X):
        dataset = WineReviewDataset(X, [0] * len(X), self.tokenizer,
                                    max_length=self.max_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size,
                                shuffle=False)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for batch in dataloader:
                # Exclude "labels" during prediction.
                inputs = {
                    k: v.to(
                        self.device
                    ) for k, v in batch.items() if k != "labels"
                }
                outputs = self.model(**inputs)
                logits = outputs.logits
                batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
                preds.extend(batch_preds)
        return np.array(preds)

    def predict_proba(self, X):
        # Returns class probabilities via softmax.
        dataset = WineReviewDataset(X, [0] * len(X), self.tokenizer,
                                    max_length=self.max_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size,
                                shuffle=False)
        self.model.eval()
        all_probs = []
        with torch.no_grad():
            for batch in dataloader:
                inputs = {
                    k: v.to(
                        self.device
                    ) for k, v in batch.items() if k != "labels"
                }
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.extend(probs)
        return np.array(all_probs)

    def score(self, X, y):
        # Use balanced accuracy as the score.
        from sklearn.metrics import balanced_accuracy_score
        preds = self.predict(X)
        return balanced_accuracy_score(y, preds)


# Transformer: isolate the "comment" column and fill missing values.
transformer = FunctionTransformer(lambda df: df["comment"].fillna(""),
                                  validate=False)

# Instantiate the custom BERT classifier.
bert_clf = BertSklearnClassifier(
    pretrained_model_name="bert-base-uncased",
    max_length=128,
    epochs=1,
    batch_size=16,
    learning_rate=3e-5
)

# Build the pipeline: extract text then classify using BERT.
pipe = Pipeline([
    ("text_isolator", transformer),
    ("clf", bert_clf)
])


def get_estimator():
    return pipe
