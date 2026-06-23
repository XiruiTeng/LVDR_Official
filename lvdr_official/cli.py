from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import DEFAULT_PATHS
from .data import load_feature_pair
from .diffusion import NoiseSchedule, generate_embeddings, load_diffusion_bundle, train_diffusion
from .features import (
    DEFAULT_INTERNVIDEO_MODEL,
    copy_existing_features,
    extract_hot_video_keypoints,
    extract_keypoint_features,
    extract_text_embeddings,
    extract_video_features,
)
from .mcts import plan_dataset
from .score import (
    MAX_SCORE_EPOCHS,
    compare_prediction_json,
    evaluate_prediction_json,
    predict_scores,
    train_score_model,
)
from .utils import load_tensor, resolve_device, seed_everything


def _default_checkpoint(name: str) -> Path:
    return DEFAULT_PATHS.checkpoint_dir / name


def _default_score_checkpoint(name: str) -> Path:
    return DEFAULT_PATHS.score_checkpoint_dir / name


def _add_common_device(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="cuda:0", help="Torch device, e.g. cuda:0 or cpu.")
    parser.add_argument("--seed", type=int, default=3407)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MCTD Ours/Ego-Exo-4D refactored runner")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect-sample", help="Load one sample and print tensor shapes.")
    inspect.add_argument("--name", default="iiith_soccer_051_2")
    inspect.add_argument("--video-root", type=Path, default=DEFAULT_PATHS.video_root)
    inspect.add_argument("--keypoint-root", type=Path, default=DEFAULT_PATHS.keypoint_root)
    inspect.add_argument("--embedding-root", type=Path, default=DEFAULT_PATHS.embedding_root)

    copy_f = sub.add_parser("copy-features", help="Copy existing .pt features into an output directory.")
    copy_f.add_argument("--input-root", type=Path, required=True)
    copy_f.add_argument("--output-dir", type=Path, required=True)
    copy_f.add_argument("--split-json", type=Path, default=None)
    copy_f.add_argument("--name-key", default="name")
    copy_f.add_argument("--extension", default=".pt")
    copy_f.add_argument("--max-samples", type=int, default=0)
    copy_f.add_argument("--skip-missing", action="store_true")
    copy_f.add_argument("--overwrite", action="store_true")

    text = sub.add_parser("extract-text", help="Extract InternVideo language-model text embeddings.")
    text.add_argument("--split-json", type=Path, required=True)
    text.add_argument("--output-dir", type=Path, required=True)
    text.add_argument("--text-key", default="comment")
    text.add_argument("--name-key", default="name")
    text.add_argument("--model-path", default=DEFAULT_INTERNVIDEO_MODEL)
    text.add_argument("--max-length", type=int, default=512)
    text.add_argument("--max-samples", type=int, default=0)
    text.add_argument("--device-map", default="auto")
    text.add_argument("--overwrite", action="store_true")

    video = sub.add_parser("extract-video", help="Extract InternVideo visual features from raw videos.")
    video.add_argument("--video-root", type=Path, required=True)
    video.add_argument("--output-dir", type=Path, required=True)
    video.add_argument("--split-json", type=Path, default=None)
    video.add_argument("--name-key", default="name")
    video.add_argument("--video-ext", default=".mp4")
    video.add_argument("--model-path", default=DEFAULT_INTERNVIDEO_MODEL)
    video.add_argument("--num-segments", type=int, default=48)
    video.add_argument("--input-size", type=int, default=448)
    video.add_argument("--max-num", type=int, default=1)
    video.add_argument("--max-samples", type=int, default=0)
    video.add_argument("--device-map", default="auto")
    video.add_argument("--overwrite", action="store_true")

    hot = sub.add_parser("extract-video-keypoints-hot", help="Use a HoT checkout to extract 3D keypoints from raw videos.")
    hot.add_argument("--hot-root", type=Path, required=True, help="Path to the cloned NationalGAILab/HoT repository.")
    hot.add_argument("--video-root", type=Path, required=True)
    hot.add_argument("--output-dir", type=Path, required=True)
    hot.add_argument("--split-json", type=Path, default=None)
    hot.add_argument("--name-key", default="name")
    hot.add_argument("--video-ext", default=".mp4")
    hot.add_argument("--python", default="python", help="Python executable inside the HoT environment.")
    hot.add_argument("--gpu", default="0")
    hot.add_argument("--max-samples", type=int, default=0)
    hot.add_argument("--overwrite", action="store_true")
    hot.add_argument("--copy-video", action="store_true", help="Copy videos into HoT demo/video instead of symlinking.")
    hot.add_argument("--fix-z", action="store_true")

    keypoint = sub.add_parser("extract-keypoint", help="Extract or copy 20x40 keypoint features.")
    keypoint.add_argument("--input-root", type=Path, required=True)
    keypoint.add_argument("--output-dir", type=Path, required=True)
    keypoint.add_argument("--split-json", type=Path, default=None)
    keypoint.add_argument("--name-key", default="name")
    keypoint.add_argument("--npz-key", default="reconstruction")
    keypoint.add_argument("--segments", type=int, default=20)
    keypoint.add_argument("--max-samples", type=int, default=0)
    keypoint.add_argument("--skip-missing", action="store_true")
    keypoint.add_argument("--overwrite", action="store_true")

    predict = sub.add_parser("predict-scores", help="Run the score model on cached diffusion embeddings.")
    _add_common_device(predict)
    predict.add_argument("--split-json", type=Path, default=DEFAULT_PATHS.test_json)
    predict.add_argument("--embedding-root", type=Path, default=DEFAULT_PATHS.embedding_root)
    predict.add_argument("--video-root", type=Path, default=DEFAULT_PATHS.video_root)
    predict.add_argument("--predict-checkpoint", type=Path, default=_default_score_checkpoint("predict_model.pt"))
    predict.add_argument("--output", type=Path, default=DEFAULT_PATHS.project_root / "outputs" / "pred_scores.json")
    predict.add_argument("--batch-size", type=int, default=8)
    predict.add_argument("--max-samples", type=int, default=0)
    predict.add_argument("--compare-json", type=Path, default=None)

    evaluate = sub.add_parser("evaluate", help="Compute rho/RL2/MSE/MAE for a prediction JSON.")
    evaluate.add_argument("--pred-json", type=Path, required=True)
    evaluate.add_argument("--pred-key", default="pred_score")
    evaluate.add_argument("--target-key", default="score")

    compare = sub.add_parser("compare", help="Compare two prediction JSON files by sample name.")
    compare.add_argument("--lhs-json", type=Path, required=True)
    compare.add_argument("--rhs-json", type=Path, required=True)
    compare.add_argument("--pred-key", default="pred_score")

    gen = sub.add_parser("generate-embeddings", help="Generate diffusion embeddings from video/keypoint features.")
    _add_common_device(gen)
    gen.add_argument("--split-json", type=Path, default=DEFAULT_PATHS.test_json)
    gen.add_argument("--video-root", type=Path, default=DEFAULT_PATHS.video_root)
    gen.add_argument("--keypoint-root", type=Path, default=DEFAULT_PATHS.keypoint_root)
    gen.add_argument("--output-root", type=Path, required=True)
    gen.add_argument("--diffusion-checkpoint", type=Path, default=_default_checkpoint("Diffusion.pt"))
    gen.add_argument(
        "--reshape-keypoint-checkpoint",
        type=Path,
        default=_default_checkpoint("reshape_keypoint_module.pt"),
    )
    gen.add_argument(
        "--reshape-all-checkpoint",
        type=Path,
        default=_default_checkpoint("reshape_all_module.pt"),
    )
    gen.add_argument("--schedule", choices=["linear", "cosine"], default="cosine")
    gen.add_argument("--max-samples", type=int, default=0)
    gen.add_argument("--save-steps", action="store_true")

    train_s = sub.add_parser("train-score", help="Train the score predictor on cached embeddings.")
    _add_common_device(train_s)
    train_s.add_argument("--train-json", type=Path, default=DEFAULT_PATHS.train_json)
    train_s.add_argument("--test-json", type=Path, default=DEFAULT_PATHS.test_json)
    train_s.add_argument("--embedding-root", type=Path, default=DEFAULT_PATHS.embedding_root)
    train_s.add_argument("--video-root", type=Path, default=DEFAULT_PATHS.video_root)
    train_s.add_argument("--output-checkpoint", type=Path, required=True)
    train_s.add_argument("--epochs", type=int, default=MAX_SCORE_EPOCHS)
    train_s.add_argument("--batch-size", type=int, default=8)
    train_s.add_argument("--eval-batch-size", type=int, default=16)
    train_s.add_argument("--lr", type=float, default=1e-4)
    train_s.add_argument("--loss-scale", type=float, default=1.0)
    train_s.add_argument("--max-samples", type=int, default=0)
    train_s.add_argument("--eval-max-samples", type=int, default=0)
    train_s.add_argument("--save-best", action="store_true")

    train_d = sub.add_parser("train-diffusion", help="Train the diffusion model.")
    _add_common_device(train_d)
    train_d.add_argument("--train-json", type=Path, default=DEFAULT_PATHS.train_json)
    train_d.add_argument("--text-root", type=Path, default=DEFAULT_PATHS.text_root)
    train_d.add_argument("--video-root", type=Path, default=DEFAULT_PATHS.video_root)
    train_d.add_argument("--keypoint-root", type=Path, default=DEFAULT_PATHS.keypoint_root)
    train_d.add_argument("--output-dir", type=Path, required=True)
    train_d.add_argument("--schedule", choices=["linear", "cosine"], default="cosine")
    train_d.add_argument("--epochs", type=int, default=300)
    train_d.add_argument("--batch-size", type=int, default=8)
    train_d.add_argument("--lr", type=float, default=1e-4)
    train_d.add_argument("--max-samples", type=int, default=0)
    train_d.add_argument("--save-every", type=int, default=10)

    plan = sub.add_parser("plan-mcts", help="Run stepwise MCTS planning from cached denoising states.")
    _add_common_device(plan)
    plan.add_argument("--split-json", type=Path, default=DEFAULT_PATHS.test_json)
    plan.add_argument("--pred-score-json", type=Path, required=True)
    plan.add_argument("--video-root", type=Path, default=DEFAULT_PATHS.video_root)
    plan.add_argument("--keypoint-root", type=Path, default=DEFAULT_PATHS.keypoint_root)
    plan.add_argument("--denoise-root", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--diffusion-checkpoint", type=Path, default=_default_checkpoint("Diffusion.pt"))
    plan.add_argument(
        "--reshape-keypoint-checkpoint",
        type=Path,
        default=_default_checkpoint("reshape_keypoint_module.pt"),
    )
    plan.add_argument(
        "--reshape-all-checkpoint",
        type=Path,
        default=_default_checkpoint("reshape_all_module.pt"),
    )
    plan.add_argument("--predict-checkpoint", type=Path, default=_default_score_checkpoint("predict_model.pt"))
    plan.add_argument("--schedule", choices=["linear", "cosine"], default="cosine")
    plan.add_argument("--num-simulations", type=int, default=150)
    plan.add_argument("--max-samples", type=int, default=0)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "inspect-sample":
        video = load_tensor(args.video_root / f"{args.name}.pt")
        keypoint = load_tensor(args.keypoint_root / f"{args.name}.pt")
        embedding_path = args.embedding_root / f"{args.name}.pt"
        payload = {
            "name": args.name,
            "video_shape": list(video.shape),
            "keypoint_shape": list(keypoint.shape),
        }
        if embedding_path.exists():
            payload["embedding_shape"] = list(load_tensor(embedding_path).shape)
        print(json.dumps(payload, indent=2))
        return

    if args.command == "copy-features":
        written = copy_existing_features(
            args.input_root,
            args.output_dir,
            split_json=args.split_json,
            name_key=args.name_key,
            extension=args.extension,
            max_samples=args.max_samples,
            skip_existing=not args.overwrite,
            skip_missing=args.skip_missing,
        )
        print(json.dumps({"num_written": len(written), "output_dir": str(args.output_dir)}, indent=2))
        return

    if args.command == "extract-text":
        written = extract_text_embeddings(
            args.split_json,
            args.output_dir,
            text_key=args.text_key,
            name_key=args.name_key,
            model_path=args.model_path,
            max_length=args.max_length,
            max_samples=args.max_samples,
            skip_existing=not args.overwrite,
            device_map=args.device_map,
        )
        print(json.dumps({"num_written": len(written), "output_dir": str(args.output_dir)}, indent=2))
        return

    if args.command == "extract-video":
        written = extract_video_features(
            args.video_root,
            args.output_dir,
            split_json=args.split_json,
            name_key=args.name_key,
            video_ext=args.video_ext,
            model_path=args.model_path,
            num_segments=args.num_segments,
            input_size=args.input_size,
            max_num=args.max_num,
            max_samples=args.max_samples,
            skip_existing=not args.overwrite,
            device_map=args.device_map,
        )
        print(json.dumps({"num_written": len(written), "output_dir": str(args.output_dir)}, indent=2))
        return

    if args.command == "extract-video-keypoints-hot":
        written = extract_hot_video_keypoints(
            args.hot_root,
            args.video_root,
            args.output_dir,
            split_json=args.split_json,
            name_key=args.name_key,
            video_ext=args.video_ext,
            python_executable=args.python,
            gpu=args.gpu,
            max_samples=args.max_samples,
            skip_existing=not args.overwrite,
            copy_video=args.copy_video,
            fix_z=args.fix_z,
        )
        print(json.dumps({"num_written": len(written), "output_dir": str(args.output_dir)}, indent=2))
        return

    if args.command == "extract-keypoint":
        written = extract_keypoint_features(
            args.input_root,
            args.output_dir,
            split_json=args.split_json,
            name_key=args.name_key,
            npz_key=args.npz_key,
            segments=args.segments,
            max_samples=args.max_samples,
            skip_existing=not args.overwrite,
            skip_missing=args.skip_missing,
        )
        print(json.dumps({"num_written": len(written), "output_dir": str(args.output_dir)}, indent=2))
        return

    if hasattr(args, "seed"):
        seed_everything(args.seed)
    device = resolve_device(args.device) if hasattr(args, "device") else torch.device("cpu")

    if args.command == "predict-scores":
        result = predict_scores(
            args.split_json,
            args.embedding_root,
            args.video_root,
            args.predict_checkpoint,
            args.output,
            device,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
        )
        if args.compare_json:
            result["compare"] = compare_prediction_json(args.output, args.compare_json)
        print(json.dumps(result, indent=2))
        return

    if args.command == "evaluate":
        print(json.dumps(evaluate_prediction_json(args.pred_json, args.pred_key, args.target_key), indent=2))
        return

    if args.command == "compare":
        print(json.dumps(compare_prediction_json(args.lhs_json, args.rhs_json, args.pred_key), indent=2))
        return

    if args.command == "generate-embeddings":
        diffusion_model, reshape_keypoint, reshape_all = load_diffusion_bundle(
            args.diffusion_checkpoint,
            args.reshape_keypoint_checkpoint,
            args.reshape_all_checkpoint,
            device,
        )
        written = generate_embeddings(
            args.split_json,
            args.video_root,
            args.keypoint_root,
            args.output_root,
            diffusion_model,
            reshape_keypoint,
            reshape_all,
            NoiseSchedule(kind=args.schedule),
            device,
            max_samples=args.max_samples,
            save_steps=args.save_steps,
        )
        print(json.dumps({"num_written": len(written), "output_root": str(args.output_root)}, indent=2))
        return

    if args.command == "train-score":
        train_score_model(
            args.train_json,
            args.test_json,
            args.embedding_root,
            args.video_root,
            args.output_checkpoint,
            device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            loss_scale=args.loss_scale,
            seed=args.seed,
            eval_batch_size=args.eval_batch_size,
            max_samples=args.max_samples,
            eval_max_samples=args.eval_max_samples,
            save_best=args.save_best,
        )
        return

    if args.command == "train-diffusion":
        train_diffusion(
            args.train_json,
            args.text_root,
            args.video_root,
            args.keypoint_root,
            args.output_dir,
            device,
            NoiseSchedule(kind=args.schedule),
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            max_samples=args.max_samples,
            save_every=args.save_every,
        )
        return

    if args.command == "plan-mcts":
        results = plan_dataset(
            args.split_json,
            args.pred_score_json,
            args.video_root,
            args.keypoint_root,
            args.denoise_root,
            args.output,
            args.diffusion_checkpoint,
            args.reshape_keypoint_checkpoint,
            args.reshape_all_checkpoint,
            args.predict_checkpoint,
            device,
            NoiseSchedule(kind=args.schedule),
            max_samples=args.max_samples,
            num_simulations=args.num_simulations,
        )
        print(json.dumps({"num_samples": len(results), "output": str(args.output)}, indent=2))
        return

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
