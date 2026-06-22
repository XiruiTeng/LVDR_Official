from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import ScoreDataset
from .metrics import score_metrics
from .models import PredictModel
from .utils import ensure_dir, limit_items, load_json, load_state_dict, save_json, seed_everything

MAX_SCORE_EPOCHS = 160


def load_predict_model(checkpoint: str | Path, device: torch.device) -> PredictModel:
    model = PredictModel(input_shape=48 * 4096)
    model.load_state_dict(load_state_dict(checkpoint))
    return model.to(device).eval()


def _prepare_embedding(embedding: torch.Tensor, device: torch.device) -> torch.Tensor:
    embedding = embedding.to(device=device, dtype=torch.float32)
    if embedding.ndim == 4 and embedding.shape[1] == 1:
        embedding = embedding.squeeze(1)
    return embedding


@torch.no_grad()
def predict_scores(
    split_json: str | Path,
    embedding_root: str | Path,
    video_root: str | Path,
    checkpoint: str | Path,
    output_json: str | Path,
    device: torch.device,
    batch_size: int = 8,
    max_samples: int | None = None,
) -> dict[str, Any]:
    dataset = ScoreDataset(split_json, embedding_root, video_root)
    if max_samples is not None and max_samples > 0:
        dataset.items = limit_items(dataset.items, max_samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = load_predict_model(checkpoint, device)

    results: list[dict[str, Any]] = []
    preds: list[float] = []
    targets: list[float] = []
    for batch in tqdm(loader, desc="predicting scores"):
        embedding = _prepare_embedding(batch["embedding"], device)
        visual = batch["visual"].to(device=device, dtype=torch.float32)
        pred_score = model(embedding, visual).detach().cpu().flatten().tolist()
        scores = batch["score"].detach().cpu().flatten().tolist()
        names = list(batch["name"])
        for name, target, pred in zip(names, scores, pred_score):
            results.append({"name": name, "score": float(target), "pred_score": float(pred)})
            preds.append(float(pred))
            targets.append(float(target))

    metrics = score_metrics(preds, targets) if preds else {"rho": 0.0, "rl2": 0.0, "mse": 0.0, "mae": 0.0}
    save_json(results, output_json)
    return {"output": str(output_json), "num_samples": len(results), **metrics}


def evaluate_prediction_json(
    pred_json: str | Path,
    pred_key: str = "pred_score",
    target_key: str = "score",
) -> dict[str, float]:
    items = load_json(pred_json)
    preds = [float(item[pred_key]) for item in items]
    targets = [float(item[target_key]) for item in items]
    return score_metrics(preds, targets)


def compare_prediction_json(
    lhs_json: str | Path,
    rhs_json: str | Path,
    pred_key: str = "pred_score",
) -> dict[str, float | int]:
    lhs = {item["name"]: float(item[pred_key]) for item in load_json(lhs_json)}
    rhs = {item["name"]: float(item[pred_key]) for item in load_json(rhs_json)}
    common = sorted(set(lhs) & set(rhs))
    if not common:
        raise ValueError("No common sample names to compare.")
    diffs = [abs(lhs[name] - rhs[name]) for name in common]
    return {
        "num_common": len(common),
        "max_abs_diff": float(max(diffs)),
        "mean_abs_diff": float(sum(diffs) / len(diffs)),
    }


def train_score_model(
    train_json: str | Path,
    test_json: str | Path,
    embedding_root: str | Path,
    video_root: str | Path,
    output_checkpoint: str | Path,
    device: torch.device,
    epochs: int = MAX_SCORE_EPOCHS,
    batch_size: int = 8,
    lr: float = 1e-4,
    loss_scale: float = 1.0,
    seed: int = 3407,
    eval_batch_size: int | None = None,
    max_samples: int | None = None,
    eval_max_samples: int | None = None,
    save_best: bool = False,
) -> None:
    if epochs > MAX_SCORE_EPOCHS:
        raise ValueError(f"score predictor epochs must be <= {MAX_SCORE_EPOCHS}, got {epochs}")

    seed_everything(seed)
    ensure_dir(Path(output_checkpoint).parent)
    train_dataset = ScoreDataset(train_json, embedding_root, video_root)
    test_dataset = ScoreDataset(test_json, embedding_root, video_root)
    if max_samples is not None and max_samples > 0:
        train_dataset.items = limit_items(train_dataset.items, max_samples)
    if eval_max_samples is not None and eval_max_samples > 0:
        test_dataset.items = limit_items(test_dataset.items, eval_max_samples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size or batch_size,
        shuffle=False,
    )
    model = PredictModel(input_shape=48 * 4096).to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    best_rho = float("-inf")

    for epoch in range(epochs):
        running_loss = 0.0
        for batch in train_loader:
            target = batch["score"].to(device=device, dtype=torch.float32).view(-1, 1)
            embedding = _prepare_embedding(batch["embedding"], device)
            visual = batch["visual"].to(device=device, dtype=torch.float32)
            pred = model(embedding, visual)
            loss = mse_loss(pred, target) * loss_scale
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())

        if (epoch + 1) % 10 == 0:
            preds: list[float] = []
            targets: list[float] = []
            model.eval()
            with torch.no_grad():
                for batch in test_loader:
                    embedding = _prepare_embedding(batch["embedding"], device)
                    visual = batch["visual"].to(device=device, dtype=torch.float32)
                    pred = model(embedding, visual).detach().cpu().flatten().tolist()
                    target = batch["score"].detach().cpu().flatten().tolist()
                    preds.extend(float(x) for x in pred)
                    targets.extend(float(x) for x in target)
            metrics = score_metrics(preds, targets)
            print(
                f"epoch={epoch + 1} loss={running_loss / max(len(train_loader), 1):.6f} "
                f"rho={metrics['rho']:.6f} rl2={metrics['rl2']:.6f}"
            )
            if save_best and metrics["rho"] > best_rho:
                best_rho = metrics["rho"]
                torch.save(model.state_dict(), output_checkpoint)
            model.train()
    if save_best and best_rho == float("-inf"):
        torch.save(model.state_dict(), output_checkpoint)
    if not save_best:
        torch.save(model.state_dict(), output_checkpoint)
