from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def score_metrics(predictions: list[float], targets: list[float]) -> dict[str, float]:
    y_pred = np.asarray(predictions, dtype=np.float64)
    y_true = np.asarray(targets, dtype=np.float64)
    if y_pred.shape != y_true.shape:
        raise ValueError(f"Shape mismatch: predictions {y_pred.shape}, targets {y_true.shape}")

    rho, _ = spearmanr(y_pred, y_true)
    if np.isnan(rho):
        rho = 0.0

    data_range = float(np.max(y_true) - np.min(y_true))
    if data_range == 0:
        rl2 = 0.0
    else:
        rl2 = float(np.mean(((y_pred - y_true) ** 2) / (data_range**2)))

    mse = float(np.mean((y_pred - y_true) ** 2))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    return {"rho": float(rho), "rl2": rl2, "mse": mse, "mae": mae}
