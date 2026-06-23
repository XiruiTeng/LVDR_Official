from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import load_json, save_json

DEFAULT_QWEN_VL_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

QWEN_COMMENT_PROMPTS: dict[str, str] = {
    "fitness": (
        "You are a professional fitness coach. Based on the following fitness exercise video, "
        "provide a detailed assessment of the quality of the performed movement.Evaluate the "
        "subject’s form, posture, control, stability, range of motion, and overall execution. "
        "Identify what was done well and what could be improved. Please provide a comprehensive "
        "paragraph of feedback, avoiding premature summarization or cutoff.Please make your "
        "response detailed, less than 200 words."
    ),
    "finediving": (
        "You are a professional finediving coach. Based on the following finediving video, "
        "provide a detailed assessment of the quality of the performed movement.Evaluate the "
        "subject’s form, posture, control, stability, range of motion, and overall execution. "
        "Identify what was done well and what could be improved. Please provide a comprehensive "
        "paragraph of feedback, avoiding premature summarization or cutoff.Please make your "
        "response detailed, less than 200 words."
    ),
    "jigsaw": (
        "You are an experienced surgical instructor. Based on the following surgical procedure "
        "log, including the sequence of gestures and their durations, provide a professional "
        "evaluation of the operator's performance. Please identify the strengths and weaknesses "
        "observed during the procedure, and share your overall assessment of the operator’s skill "
        "level. Your response should be written as a natural, continuous paragraph — do not use "
        "bullet points or score tables. Use appropriate medical and surgical terminology, and "
        "adopt a tone similar to that of expert feedback during surgical training. Please provide "
        "a comprehensive paragraph of feedback, avoiding premature summarization or cutoff.Please "
        "make your response detailed, less than 200 words."
    ),
    "cataract": (
        "You are an experienced ophthalmic surgical instructor. Based on the following cataract "
        "surgery procedure log, which includes the sequence of gestures, instruments used, and "
        "their durations, provide a detailed professional evaluation of the surgeon’s performance. "
        "Discuss the technical execution, efficiency, instrument handling, tissue management, and "
        "overall flow of the procedure. Identify specific strengths and weaknesses observed "
        "throughout the operation, commenting on aspects such as precision, consistency, and "
        "economy of motion. Conclude with a holistic assessment of the surgeon’s current skill "
        "level and readiness for independent practice. Write your response as a continuous, "
        "cohesive paragraph — do not use bullet points or scoring tables. Use appropriate surgical "
        "terminology and maintain the tone of constructive expert feedback typically used in "
        "advanced surgical training. Your response should be comprehensive and insightful, within "
        "150–200 words."
    ),
}


def _load_json_list(path: str | Path) -> list[dict[str, Any]]:
    items = load_json(path)
    if not isinstance(items, list):
        raise TypeError(f"Expected a list in {path}, got {type(items)!r}")
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise TypeError(f"Expected item {index} in {path} to be a dict, got {type(item)!r}")
    return items


def _unique_names(items: list[dict[str, Any]], name_key: str) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for item in items:
        name = str(item[name_key])
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _build_text_lookup(
    items: list[dict[str, Any]],
    name_key: str,
    source_text_key: str,
) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for item in items:
        name = str(item[name_key])
        if source_text_key not in item:
            raise KeyError(f"Missing text key {source_text_key!r} for sample {name!r}.")
        text = str(item[source_text_key])
        if name in lookup and lookup[name] != text:
            raise ValueError(f"Found conflicting comments for sample {name!r}.")
        lookup[name] = text
    return lookup


def _build_value_lookup(
    items: list[dict[str, Any]],
    name_key: str,
    value_key: str,
) -> dict[str, Any]:
    lookup: dict[str, Any] = {}
    for item in items:
        name = str(item[name_key])
        if value_key not in item:
            raise KeyError(f"Missing key {value_key!r} for sample {name!r}.")
        value = item[value_key]
        if name in lookup and lookup[name] != value:
            raise ValueError(f"Found conflicting {value_key!r} values for sample {name!r}.")
        lookup[name] = value
    return lookup


def _list_video_paths(video_root: str | Path, video_ext: str, recursive: bool, include_glob: str | None) -> list[Path]:
    root = Path(video_root)
    pattern = include_glob or f"*{video_ext}"
    paths = root.rglob(pattern) if recursive else root.glob(pattern)
    return sorted(path for path in paths if path.is_file())


def _sample_name_from_video_path(path: Path, strip_suffix: str | None = None) -> str:
    if strip_suffix:
        filename = path.name
        if filename.endswith(strip_suffix):
            return filename[: -len(strip_suffix)]
        stem = path.stem
        if stem.endswith(strip_suffix):
            return stem[: -len(strip_suffix)]
    return path.stem


def _resolve_video_path(video_root: str | Path, name: str, video_ext: str, recursive: bool) -> Path | None:
    direct = Path(video_root) / f"{name}{video_ext}"
    if direct.exists():
        return direct
    if recursive:
        matches = sorted(Path(video_root).rglob(f"{name}{video_ext}"))
        for match in matches:
            if match.is_file():
                return match
    return None


def _load_existing_output(path: str | Path, name_key: str) -> list[dict[str, Any]]:
    output_path = Path(path)
    if not output_path.exists():
        return []
    return _load_json_list(output_path)


def _feature_exists(root: str | Path | None, name: str, extension: str) -> bool:
    if root is None:
        return True
    return (Path(root) / f"{name}{extension}").exists()


def available_qwen_prompt_presets() -> list[str]:
    return sorted(QWEN_COMMENT_PROMPTS)


def generate_qwen_video_comments(
    video_root: str | Path,
    output_json: str | Path,
    split_json: str | Path | None = None,
    name_key: str = "name",
    output_text_key: str = "comment",
    video_ext: str = ".mp4",
    recursive: bool = False,
    include_glob: str | None = None,
    strip_suffix: str | None = None,
    model_id: str = DEFAULT_QWEN_VL_MODEL,
    prompt_preset: str = "fitness",
    prompt: str | None = None,
    max_pixels: int = 360 * 420,
    fps: float = 1.0,
    max_new_tokens: int = 128,
    device: str = "cuda",
    device_map: str = "auto",
    attn_implementation: str | None = None,
    max_samples: int | None = None,
    save_every: int = 1,
    resume: bool = False,
    skip_missing: bool = False,
) -> dict[str, Any]:
    """Generate per-video comments with Qwen2.5-VL and save the JSON used by text extraction."""
    if prompt is None:
        if prompt_preset not in QWEN_COMMENT_PROMPTS:
            choices = ", ".join(available_qwen_prompt_presets())
            raise KeyError(f"Unknown prompt preset {prompt_preset!r}. Available presets: {choices}")
        prompt = QWEN_COMMENT_PROMPTS[prompt_preset]

    if split_json is None:
        video_paths = _list_video_paths(video_root, video_ext=video_ext, recursive=recursive, include_glob=include_glob)
        samples = [(_sample_name_from_video_path(path, strip_suffix=strip_suffix), path) for path in video_paths]
    else:
        names = _unique_names(_load_json_list(split_json), name_key=name_key)
        samples = []
        for name in names:
            path = _resolve_video_path(video_root, name=name, video_ext=video_ext, recursive=recursive)
            if path is None:
                if skip_missing:
                    continue
                raise FileNotFoundError(f"Cannot find video for sample {name!r} under {video_root}.")
            samples.append((name, path))

    if max_samples and max_samples > 0:
        samples = samples[:max_samples]
    if not samples:
        raise ValueError(f"No videos found under {video_root}.")

    output_items = _load_existing_output(output_json, name_key=name_key) if resume else []
    done_names = {str(item[name_key]) for item in output_items if name_key in item}

    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model_kwargs: dict[str, Any] = {"torch_dtype": "auto", "device_map": device_map}
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
    processor = AutoProcessor.from_pretrained(model_id)

    generated = 0
    with torch.no_grad():
        for name, video_path in samples:
            if name in done_names:
                continue
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": str(video_path),
                            "max_pixels": max_pixels,
                            "fps": fps,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(device)
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, eos_token_id=None)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            output_items.append({name_key: name, output_text_key: output_text[0]})
            done_names.add(name)
            generated += 1
            if save_every > 0 and generated % save_every == 0:
                save_json(output_items, output_json)

    save_json(output_items, output_json)
    return {
        "num_total": len(samples),
        "num_generated": generated,
        "num_written": len(output_items),
        "output_json": str(output_json),
    }


def generate_video_comments(
    source_json: str | Path,
    output_json: str | Path,
    split_json: str | Path | None = None,
    name_key: str = "name",
    source_text_key: str = "commentary",
    output_text_key: str = "comment",
    score_json: str | Path | None = None,
    score_key: str = "score",
    output_score_key: str = "score",
    max_samples: int | None = None,
    skip_missing: bool = False,
) -> dict[str, Any]:
    """Convert ground-truth commentary JSON into video_comment JSON used by text embedding extraction."""
    source_items = _load_json_list(source_json)
    text_lookup = _build_text_lookup(source_items, name_key=name_key, source_text_key=source_text_key)

    if split_json is None:
        names = _unique_names(source_items, name_key=name_key)
    else:
        split_items = _load_json_list(split_json)
        names = _unique_names(split_items, name_key=name_key)
    ordered_names = names[:max_samples] if max_samples and max_samples > 0 else names

    score_lookup = None
    if score_json is not None:
        score_lookup = _build_value_lookup(_load_json_list(score_json), name_key=name_key, value_key=score_key)

    output_items: list[dict[str, Any]] = []
    missing_comments: list[str] = []
    missing_scores: list[str] = []
    for name in ordered_names:
        if name not in text_lookup:
            missing_comments.append(name)
            if skip_missing:
                continue
            raise KeyError(f"No comment found for sample {name!r}.")

        output_item: dict[str, Any] = {
            name_key: name,
            output_text_key: text_lookup[name],
        }
        if score_lookup is not None:
            if name not in score_lookup:
                missing_scores.append(name)
                if skip_missing:
                    continue
                raise KeyError(f"No score found for sample {name!r}.")
            output_item[output_score_key] = score_lookup[name]
        output_items.append(output_item)

    save_json(output_items, output_json)
    return {
        "num_written": len(output_items),
        "output_json": str(output_json),
        "num_missing_comments": len(missing_comments),
        "num_missing_scores": len(missing_scores),
    }


def generate_text_ground_truth_from_pairs(
    pair_json: str | Path,
    output_json: str | Path,
    first_name_key: str = "video_0_name",
    first_comment_key: str = "video_0_comment",
    second_name_key: str = "video_1_name",
    second_comment_key: str = "video_1_comment",
    output_name_key: str = "name",
    output_text_key: str = "commentary",
    existing_feature_root: str | Path | None = None,
    existing_feature_ext: str = ".pt",
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Extract one text comment per video from pairwise action-difference data."""
    pair_items = _load_json_list(pair_json)
    output_items: list[dict[str, Any]] = []
    seen: set[str] = set()
    skipped_missing_features = 0

    for item in pair_items:
        for name_key, comment_key in (
            (first_name_key, first_comment_key),
            (second_name_key, second_comment_key),
        ):
            if name_key not in item:
                raise KeyError(f"Missing key {name_key!r} in pair item.")
            if comment_key not in item:
                raise KeyError(f"Missing key {comment_key!r} in pair item.")

            name = str(item[name_key])
            comment = str(item[comment_key])
            if name in seen:
                continue
            if not _feature_exists(existing_feature_root, name, existing_feature_ext):
                skipped_missing_features += 1
                continue
            seen.add(name)
            output_items.append({output_name_key: name, output_text_key: comment})
            if max_samples and max_samples > 0 and len(output_items) >= max_samples:
                save_json(output_items, output_json)
                return {
                    "num_written": len(output_items),
                    "num_skipped_missing_features": skipped_missing_features,
                    "output_json": str(output_json),
                }

    save_json(output_items, output_json)
    return {
        "num_written": len(output_items),
        "num_skipped_missing_features": skipped_missing_features,
        "output_json": str(output_json),
    }
