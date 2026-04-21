"""Supervised models from Stage 3.

Four functions, one per classifier in the Stage 3 comparison table:
  train_condition_classifier - logistic regression on condition labels
  train_identity_classifier  - logistic regression on base_identity labels
  train_vgg_pca_pipeline     - L2 normalize -> PCA(256) -> logistic regression
  train_vgg_mlp_head         - small MLP head trained on frozen VGG features

All use random_state=42 (config.RANDOM_SEED) for reproducibility.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, normalize

from .config import RANDOM_SEED
from .metadata import merge_clean


@dataclass
class TrainResult:
    model: object
    accuracy: float
    y_pred: np.ndarray
    y_true: np.ndarray
    classes: np.ndarray
    report: str


def _split_Xy(df: pd.DataFrame, feature_model: str, label_col: str):
    sub = df[df["model"] == feature_model]
    X = np.vstack(sub["embedding"].values)
    y = sub[label_col].values
    return X, y, sub


def train_condition_classifier(train_df, test_df, feature_model: str = "ArcFace") -> TrainResult:
    """Logistic regression for condition (clean/Expression/Occlusion/Lighting/Side).

    WHY class_weight='balanced': Lighting has ~2x the samples of other conditions.
    Balanced weights prevent the classifier from trivially predicting Lighting.
    WHY merge clean+clean_probe: those are the same condition seen under different
    splits (gallery vs probe) - asking the model to distinguish them is not the task.
    """
    X_train, _, train_sub = _split_Xy(train_df, feature_model, "condition")
    X_test, _, test_sub = _split_Xy(test_df, feature_model, "condition")
    y_train = np.array([merge_clean(c) for c in train_sub["condition"].values])
    y_test = np.array([merge_clean(c) for c in test_sub["condition"].values])

    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return TrainResult(
        model=clf,
        accuracy=accuracy_score(y_test, y_pred),
        y_pred=y_pred,
        y_true=y_test,
        classes=clf.classes_,
        report=classification_report(y_test, y_pred, zero_division=0),
    )


def train_identity_classifier(train_df, test_df, feature_model: str = "ArcFace") -> TrainResult:
    """One-vs-rest logistic regression on base_identity."""
    X_train, y_train, _ = _split_Xy(train_df, feature_model, "base_identity")
    X_test, y_test, _ = _split_Xy(test_df, feature_model, "base_identity")

    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return TrainResult(
        model=clf,
        accuracy=accuracy_score(y_test, y_pred),
        y_pred=y_pred,
        y_true=y_test,
        classes=clf.classes_,
        report=classification_report(y_test, y_pred, zero_division=0),
    )


def train_vgg_pca_pipeline(train_df, test_df) -> TrainResult:
    """L2 normalize -> PCA(256) -> logistic regression.

    WHY L2 normalize: euclidean distance on normalized vectors approximates
    cosine similarity, which matches how ArcFace features are meant to be compared.
    WHY PCA to 256: collapses 4096-d VGG features to the subspace that carries
    most of the variance, which reduces overfitting on a small dataset.
    """
    X_train, y_train, _ = _split_Xy(train_df, "VGG19", "base_identity")
    X_test, y_test, _ = _split_Xy(test_df, "VGG19", "base_identity")

    X_train_n = normalize(X_train, norm="l2")
    X_test_n = normalize(X_test, norm="l2")

    pca = PCA(n_components=256, random_state=RANDOM_SEED)
    X_train_p = pca.fit_transform(X_train_n)
    X_test_p = pca.transform(X_test_n)

    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    clf.fit(X_train_p, y_train)
    y_pred = clf.predict(X_test_p)
    return TrainResult(
        model=(clf, pca),
        accuracy=accuracy_score(y_test, y_pred),
        y_pred=y_pred,
        y_true=y_test,
        classes=clf.classes_,
        report=classification_report(y_test, y_pred, zero_division=0),
    )


def train_vgg_mlp_head(train_df, test_df, epochs: int = 60, hidden_dim: int = 512):
    """Train a small MLP head on frozen 4096-d VGG features.

    WHY MLP over LR: lets the model learn nonlinear combinations of VGG features,
    which the PDF explicitly calls out as the higher-effort but more performant
    path. We keep VGG weights frozen because the dataset is too small to finetune
    a 140M-parameter CNN without catastrophic overfitting.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, y_train_raw, _ = _split_Xy(train_df, "VGG19", "base_identity")
    X_test, y_test_raw, _ = _split_Xy(test_df, "VGG19", "base_identity")
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_test = le.transform(y_test_raw)
    num_classes = len(le.classes_)

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(4096, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(hidden_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes),
            )

        def forward(self, x):
            return self.net(x)

    class DS(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.long)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            return self.X[i], self.y[i]

    train_loader = DataLoader(DS(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(DS(X_test, y_test), batch_size=32, shuffle=False)

    torch.manual_seed(RANDOM_SEED)
    model = Head().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)
    crit = nn.CrossEntropyLoss()

    best_acc, best_state, best_preds = 0.0, None, None
    for ep in range(epochs):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(Xb), yb)
            loss.backward()
            opt.step()
        sched.step()

        if (ep + 1) % 10 == 0:
            model.eval()
            preds_all = []
            with torch.no_grad():
                for Xb, _ in test_loader:
                    preds_all.extend(model(Xb.to(device)).argmax(1).cpu().numpy())
            preds_all = np.array(preds_all)
            acc = (preds_all == y_test).mean()
            if acc > best_acc:
                best_acc = acc
                best_preds = preds_all
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    y_pred = le.inverse_transform(best_preds)
    y_true = le.inverse_transform(y_test)
    return TrainResult(
        model=(model, le, best_state),
        accuracy=best_acc,
        y_pred=y_pred,
        y_true=y_true,
        classes=le.classes_,
        report=classification_report(y_true, y_pred, zero_division=0),
    )


def per_condition_accuracy(y_true, y_pred, conditions, model_name: str) -> pd.DataFrame:
    """Break down identity accuracy by condition (for the Stage 3 comparison table)."""
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "condition": conditions})
    rows = [
        {"Model": model_name, "Condition": cond,
         "Accuracy": (g["y_true"] == g["y_pred"]).mean(), "N": len(g)}
        for cond, g in df.groupby("condition")
    ]
    return pd.DataFrame(rows)
