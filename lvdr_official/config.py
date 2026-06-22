from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT_ROOT = PACKAGE_ROOT
DEFAULT_DATA_ROOT = DEFAULT_PROJECT_ROOT / "data"


def _path_from_env(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


@dataclass(frozen=True)
class OursPaths:
    project_root: Path = _path_from_env("LVDR_PROJECT_ROOT", DEFAULT_PROJECT_ROOT)
    dataset_root: Path = _path_from_env("OURS_DATA_ROOT", DEFAULT_DATA_ROOT)

    @property
    def train_json(self) -> Path:
        return _path_from_env("OURS_TRAIN_JSON", self.dataset_root / "splits" / "train_data_new.json")

    @property
    def test_json(self) -> Path:
        return _path_from_env("OURS_TEST_JSON", self.dataset_root / "splits" / "test_data.json")

    @property
    def video_root(self) -> Path:
        return _path_from_env("OURS_VIDEO_ROOT", self.project_root / "outputs" / "video_feature")

    @property
    def keypoint_root(self) -> Path:
        return _path_from_env("OURS_KEYPOINT_ROOT", self.project_root / "outputs" / "keypoint_feature")

    @property
    def text_root(self) -> Path:
        return _path_from_env("OURS_TEXT_ROOT", self.project_root / "outputs" / "text_embedding")

    @property
    def checkpoint_dir(self) -> Path:
        return _path_from_env("OURS_CHECKPOINT_DIR", self.project_root / "checkpoints" / "diffusion")

    @property
    def score_checkpoint_dir(self) -> Path:
        return _path_from_env(
            "OURS_SCORE_CHECKPOINT_DIR",
            self.project_root / "checkpoints" / "score",
        )

    @property
    def embedding_root(self) -> Path:
        return _path_from_env("OURS_EMBEDDING_ROOT", self.project_root / "outputs" / "diffusion_embedding")

    @property
    def linear_embedding_root(self) -> Path:
        return _path_from_env(
            "OURS_LINEAR_EMBEDDING_ROOT",
            self.project_root / "outputs" / "diffusion_embedding_linear",
        )


DEFAULT_PATHS = OursPaths()
