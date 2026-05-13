from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_score))


def safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(average_precision_score(y_true, y_score))


def false_rates(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    fpr = fp / max(1, fp + tn)
    fnr = fn / max(1, fn + tp)
    return float(fpr), float(fnr)


@torch.no_grad()
def evaluate_binary_classifier(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, float]:
    model.to(device)
    model.eval()

    all_probs = []
    all_labels = []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits).detach().cpu().numpy()

        all_probs.append(probs)
        all_labels.append(y.numpy())

    y_score = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels).astype(int)
    y_pred = (y_score >= threshold).astype(int)
    fpr, fnr = false_rates(y_true, y_pred)

    return {
        "eval_acc": float(accuracy_score(y_true, y_pred)),
        "eval_balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "eval_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "eval_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "eval_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "eval_roc_auc": safe_auc(y_true, y_score),
        "eval_auprc": safe_auprc(y_true, y_score),
        "eval_false_positive_rate": fpr,
        "eval_false_negative_rate": fnr,
        "num-examples": int(len(y_true)),
    }
