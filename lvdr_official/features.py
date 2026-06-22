from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .utils import ensure_dir, limit_items, load_json


DEFAULT_INTERNVIDEO_MODEL = "OpenGVLab/InternVideo2_5_Chat_8B"


def _load_split(split_json: str | Path | None, max_samples: int | None = None) -> list[dict[str, Any]]:
    if split_json is None:
        return []
    items = load_json(split_json)
    if not isinstance(items, list):
        raise TypeError(f"Expected a list in {split_json}, got {type(items)!r}")
    return limit_items(items, max_samples)


def _names_from_split(
    split_json: str | Path | None,
    name_key: str = "name",
    max_samples: int | None = None,
) -> list[str]:
    return [str(item[name_key]) for item in _load_split(split_json, max_samples)]


def copy_existing_features(
    input_root: str | Path,
    output_dir: str | Path,
    split_json: str | Path | None = None,
    name_key: str = "name",
    extension: str = ".pt",
    max_samples: int | None = None,
    skip_existing: bool = True,
    skip_missing: bool = False,
) -> list[Path]:
    input_root = Path(input_root)
    output_dir = ensure_dir(output_dir)
    names = _names_from_split(split_json, name_key=name_key, max_samples=max_samples)
    if not names:
        names = [path.stem for path in sorted(input_root.glob(f"*{extension}"))]
        names = names[:max_samples] if max_samples and max_samples > 0 else names

    written: list[Path] = []
    for name in tqdm(names, desc=f"copying {input_root.name}"):
        source = input_root / f"{name}{extension}"
        target = output_dir / f"{name}{extension}"
        if target.exists() and skip_existing:
            written.append(target)
            continue
        if not source.exists():
            if skip_missing:
                continue
            raise FileNotFoundError(source)
        shutil.copy2(source, target)
        written.append(target)
    return written


def extract_text_embeddings(
    split_json: str | Path,
    output_dir: str | Path,
    text_key: str = "comment",
    name_key: str = "name",
    model_path: str = DEFAULT_INTERNVIDEO_MODEL,
    max_length: int = 512,
    max_samples: int | None = None,
    skip_existing: bool = True,
    device_map: str = "auto",
) -> list[Path]:
    from transformers import AutoModel, AutoTokenizer

    output_dir = ensure_dir(output_dir)
    items = _load_split(split_json, max_samples)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device_map,
    ).eval()

    written: list[Path] = []
    for item in tqdm(items, desc="extracting text embeddings"):
        name = str(item[name_key])
        output_path = output_dir / f"{name}.pt"
        if output_path.exists() and skip_existing:
            written.append(output_path)
            continue
        text = str(item[text_key])
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        with torch.no_grad():
            outputs = model.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            last_hidden_state = outputs.hidden_states[-1].detach().cpu()
        torch.save(last_hidden_state, output_path)
        written.append(output_path)
    return written


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image,
    min_num: int = 1,
    max_num: int = 6,
    image_size: int = 448,
    use_thumbnail: bool = False,
) -> list[Any]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    }
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio,
        target_ratios,
        orig_width,
        orig_height,
        image_size,
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def get_index(
    bound: tuple[float, float] | None,
    fps: float,
    max_frame: int,
    first_idx: int = 0,
    num_segments: int = 48,
) -> np.ndarray:
    if bound:
        start, end = bound
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    return np.array(
        [
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_segments)
        ]
    )


def load_video(
    video_path: str | Path,
    bound: tuple[float, float] | None = None,
    input_size: int = 448,
    max_num: int = 1,
    num_segments: int = 48,
) -> tuple[torch.Tensor, list[int]]:
    from decord import VideoReader, cpu
    from PIL import Image

    vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())
    transform = build_transform(input_size=input_size)
    pixel_values_list, num_patches_list = [], []
    for frame_index in get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments):
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        tiles = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = torch.stack([transform(tile) for tile in tiles])
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    return torch.cat(pixel_values_list), num_patches_list


def _video_names(video_root: Path, video_ext: str, max_samples: int | None) -> list[str]:
    names = [path.stem for path in sorted(video_root.glob(f"*{video_ext}"))]
    return names[:max_samples] if max_samples and max_samples > 0 else names


def extract_video_features(
    video_root: str | Path,
    output_dir: str | Path,
    split_json: str | Path | None = None,
    name_key: str = "name",
    video_ext: str = ".mp4",
    model_path: str = DEFAULT_INTERNVIDEO_MODEL,
    num_segments: int = 48,
    input_size: int = 448,
    max_num: int = 1,
    max_samples: int | None = None,
    skip_existing: bool = True,
    device_map: str = "auto",
) -> list[Path]:
    from transformers import AutoModel

    video_root = Path(video_root)
    output_dir = ensure_dir(output_dir)
    names = _names_from_split(split_json, name_key=name_key, max_samples=max_samples)
    if not names:
        names = _video_names(video_root, video_ext, max_samples)

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device_map,
    ).eval()

    written: list[Path] = []
    for name in tqdm(names, desc="extracting video features"):
        output_path = output_dir / f"{name}.pt"
        if output_path.exists() and skip_existing:
            written.append(output_path)
            continue
        video_path = video_root / f"{name}{video_ext}"
        if not video_path.exists():
            raise FileNotFoundError(video_path)
        with torch.no_grad():
            pixel_values, _ = load_video(
                video_path,
                num_segments=num_segments,
                input_size=input_size,
                max_num=max_num,
            )
            pixel_values = pixel_values.to(model.device, dtype=dtype)
            feature = model.extract_feature(pixel_values).detach().cpu()
        torch.save(feature, output_path)
        written.append(output_path)
    return written


def get_angle(point_a: np.ndarray, point_b: np.ndarray, point_c: np.ndarray) -> float:
    ab = point_b - point_a
    ac = point_c - point_a
    denom = np.linalg.norm(ab) * np.linalg.norm(ac)
    cos_theta = np.dot(ab, ac) / denom if denom != 0 else 0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def compute_distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
    return float(np.linalg.norm(point_a - point_b))


def get_speed(point_list: list[np.ndarray], second: int) -> float:
    total_distance = sum(
        compute_distance(point_list[i], point_list[i + 1])
        for i in range(len(point_list) - 1)
    )
    return total_distance / second if second != 0 else 0


def get_position(point_list: list[np.ndarray]) -> np.ndarray:
    avg = np.mean(point_list, axis=0)
    norm = np.linalg.norm(avg)
    return avg / norm if norm != 0 else avg


def process_joint(points: list[np.ndarray], angles: list[float], second: int) -> np.ndarray:
    avg_angle = np.mean(angles) if angles else 0
    speed = get_speed(points, second)
    pos = get_position(points)
    vec = np.array([avg_angle, speed, *pos])
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec


def preprocess_keypoints(keypoint: np.ndarray, segments: int = 20) -> list[list[float]]:
    joints = ["1", "2", "4", "5", "11", "12", "14", "15"]
    angle_indices = {
        "1": (1, 0, 2),
        "2": (2, 1, 3),
        "4": (4, 0, 5),
        "5": (5, 4, 6),
        "11": (11, 8, 12),
        "12": (12, 11, 13),
        "14": (14, 8, 15),
        "15": (15, 14, 16),
    }
    total_frames = keypoint.shape[0]
    base_length = total_frames // segments
    segment_bounds = []
    current_start = 0
    for i in range(segments):
        if i < segments - 1:
            seg_length = base_length
        else:
            seg_length = total_frames - current_start
        segment_bounds.append((current_start, current_start + seg_length))
        current_start += seg_length

    final_tensor: list[list[float]] = []
    for start, end in segment_bounds:
        point_dict: dict[str, list[np.ndarray]] = {joint: [] for joint in joints}
        angle_dict: dict[str, list[float]] = {joint: [] for joint in joints}
        for frame_idx in range(start, end):
            for joint in joints:
                point_dict[joint].append(keypoint[frame_idx, int(joint)])
            for joint, (a, b, c) in angle_indices.items():
                angle_dict[joint].append(
                    get_angle(keypoint[frame_idx, a], keypoint[frame_idx, b], keypoint[frame_idx, c])
                )

        second = end - start
        joint_features = []
        for joint in joints:
            vec = process_joint(point_dict[joint], angle_dict[joint], second)
            joint_features.extend(float(x) for x in vec)
        final_tensor.append(joint_features)
    return final_tensor


def _find_keypoint_source(input_root: Path, name: str) -> Path:
    pt_path = input_root / f"{name}.pt"
    if pt_path.exists():
        return pt_path
    candidates = [
        input_root / name / "output_3D",
        input_root / name,
    ]
    for directory in candidates:
        if directory.exists():
            npz_files = sorted(directory.glob("*.npz"))
            if npz_files:
                return npz_files[0]
    npz_path = input_root / f"{name}.npz"
    if npz_path.exists():
        return npz_path
    raise FileNotFoundError(f"No .pt or .npz keypoint source found for {name} in {input_root}")


def extract_keypoint_features(
    input_root: str | Path,
    output_dir: str | Path,
    split_json: str | Path | None = None,
    name_key: str = "name",
    npz_key: str = "reconstruction",
    segments: int = 20,
    max_samples: int | None = None,
    skip_existing: bool = True,
    skip_missing: bool = False,
) -> list[Path]:
    input_root = Path(input_root)
    output_dir = ensure_dir(output_dir)
    names = _names_from_split(split_json, name_key=name_key, max_samples=max_samples)
    if not names:
        names = [path.stem for path in sorted(input_root.glob("*.pt"))]
        if not names:
            names = [path.name for path in sorted(input_root.iterdir()) if path.is_dir()]
        names = names[:max_samples] if max_samples and max_samples > 0 else names

    written: list[Path] = []
    for name in tqdm(names, desc="extracting keypoint features"):
        output_path = output_dir / f"{name}.pt"
        if output_path.exists() and skip_existing:
            written.append(output_path)
            continue
        try:
            source = _find_keypoint_source(input_root, name)
        except FileNotFoundError:
            if skip_missing:
                continue
            raise
        if source.suffix == ".pt":
            shutil.copy2(source, output_path)
        else:
            keypoint = np.load(source)[npz_key]
            non_zero_mask = ~np.all(keypoint == 0, axis=(1, 2))
            non_zero_frames = keypoint[non_zero_mask]
            joint_tensor = torch.tensor(
                preprocess_keypoints(non_zero_frames, segments=segments),
                dtype=torch.float32,
            )
            torch.save(joint_tensor, output_path)
        written.append(output_path)
    return written
