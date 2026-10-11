import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


def prepare_features(train_df, full_df, feature_cols, lower: float = 0.01, upper: float = 0.99):
    quantile_low = train_df[feature_cols].quantile(lower)
    quantile_high = train_df[feature_cols].quantile(upper)
    clipped = full_df.copy()
    clipped[feature_cols] = full_df[feature_cols].clip(quantile_low, quantile_high, axis=1)
    scaler = StandardScaler().fit(clipped.reindex(train_df.index)[feature_cols])
    scaled = clipped.copy()
    scaled[feature_cols] = scaler.transform(clipped[feature_cols])
    params = {
        "features": list(feature_cols),
        "clip_lower": quantile_low.tolist(),
        "clip_upper": quantile_high.tolist(),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }
    return scaled, params


def train_model(model, train_loader: DataLoader, epochs: int, lr: float, device, pos_weight=None):
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    model.train()
    for _ in range(epochs):
        total_loss, count = 0.0, 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(by)
            count += len(by)
        history.append(total_loss / max(count, 1))
    return history


@torch.no_grad()
def predict_proba(model, loader: DataLoader, device):
    model.eval()
    probs, labels = [], []
    for bx, by in loader:
        logits = model(bx.to(device))
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(by.numpy())
    return np.concatenate(labels), np.concatenate(probs)


def compute_metrics(y_true, y_prob, threshold: float = 0.5):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "n_samples": int(len(y_true)),
        "positive_rate": round(float(y_true.mean()), 4),
        "threshold": threshold,
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
        metrics["pr_auc"] = round(float(average_precision_score(y_true, y_prob)), 4)
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
    return metrics


def fit_pooled_scaler(train_df, feature_cols, lower: float = 0.01, upper: float = 0.99):
    quantile_low = train_df[feature_cols].quantile(lower)
    quantile_high = train_df[feature_cols].quantile(upper)
    clipped = train_df[feature_cols].clip(quantile_low, quantile_high, axis=1)
    scaler = StandardScaler().fit(clipped)
    return quantile_low, quantile_high, scaler


def transform_frame(df, feature_cols, quantile_low, quantile_high, scaler):
    out = df.copy()
    out[feature_cols] = scaler.transform(df[feature_cols].clip(quantile_low, quantile_high, axis=1))
    return out
