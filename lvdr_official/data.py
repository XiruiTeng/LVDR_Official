from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .utils import load_json, load_tensor


class DiffusionTrainingDataset(Dataset):
    def __init__(
        self,
        json_path: str | Path,
        text_root: str | Path,
        video_root: str | Path,
        keypoint_root: str | Path,
    ) -> None:
        self.items: list[dict[str, Any]] = load_json(json_path)
        self.text_root = Path(text_root)
        self.video_root = Path(video_root)
        self.keypoint_root = Path(keypoint_root)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        name = item["name"]
        text_embedding = load_tensor(self.text_root / f"{name}.pt").squeeze(0)
        return {
            "name": name,
            "score": float(item["score"]),
            "video_feature": load_tensor(self.video_root / f"{name}.pt"),
            "keypoint_feature": load_tensor(self.keypoint_root / f"{name}.pt"),
            "text_embedding": text_embedding,
        }


class ScoreDataset(Dataset):
    def __init__(
        self,
        json_path: str | Path,
        embedding_root: str | Path,
        video_root: str | Path,
    ) -> None:
        self.items: list[dict[str, Any]] = load_json(json_path)
        self.embedding_root = Path(embedding_root)
        self.video_root = Path(video_root)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        name = item["name"]
        return {
            "name": name,
            "score": float(item["score"]),
            "embedding": load_tensor(self.embedding_root / f"{name}.pt"),
            "visual": load_tensor(self.video_root / f"{name}.pt"),
        }


def load_feature_pair(
    name: str,
    video_root: str | Path,
    keypoint_root: str | Path,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    video = load_tensor(Path(video_root) / f"{name}.pt").to(device=device, dtype=torch.float32)
    keypoint = load_tensor(Path(keypoint_root) / f"{name}.pt").to(device=device, dtype=torch.float32)
    return video.unsqueeze(0), keypoint.unsqueeze(0)
