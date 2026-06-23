# LVDR Official Reproduction on Our Dataset

This repository contains an engineered reproduction of the original LVDR/MCTD code for Our Dataset. The model architecture, diffusion schedule, batch sizes, learning rates, and seed follow the original implementation unless explicitly overridden by command-line arguments.

The score predictor is capped at 160 epochs. Passing `--epochs` larger than 160 raises an error, because the reproduced run peaked around epoch 160 and later collapsed to constant predictions.

## Repository Layout

- `lvdr_official/`: Python package containing datasets, models, feature extraction, diffusion training/inference, score prediction, metrics, and MCTS utilities.
- `scripts/*.py`: Python entry points for each pipeline stage.
- `configs/our_dataset.yaml`: relative-path configuration and default hyperparameters.
- `data/`: local input data directory. Large data files are ignored by git.
- `outputs/`: generated embeddings and prediction JSON files. Ignored by git.
- `checkpoints/`: trained model checkpoints. Ignored by git.

## Environment

Install dependencies from the project root:

```text
python -m pip install -r requirements.txt
```

The original reproduction used GPU0. You can choose a GPU by passing `--device cuda:0` to training and inference commands.

## Prepare Data

Place files under these relative paths:

```text
data/splits/train_data_new.json
data/splits/test_data.json
data/text_comments/video_comment_train.json
data/text_comments/video_comment_test.json
```

If extracting features from raw data, place raw inputs here:

```text
data/raw_videos/<name>.mp4
data/raw_keypoints/<name>.npz
```

or:

```text
data/raw_keypoints/<name>/output_3D/*.npz
```

Video feature extraction reads each `<name>.mp4` and writes `outputs/video_feature/<name>.pt`.
Keypoint feature extraction reads either one sequence file at `data/raw_keypoints/<name>.npz`
or a sorted frame/clip sequence under `data/raw_keypoints/<name>/output_3D/*.npz`, then writes
`outputs/keypoint_feature/<name>.pt`. Raw keypoint `.npz` files should contain a
`reconstruction` array by default; pass `--npz-key` if your array uses another key.

If reusing processed visual/keypoint features, place them here:

```text
data/processed_features/video_feature/<name>.pt
data/processed_features/keypoint_feature/<name>.pt
```

All generated files should stay under:

```text
outputs/
checkpoints/
```

## Step 1: Extract Text Embeddings

Train comments:

```text
python scripts/extract_text_embeddings.py --split-json data/text_comments/video_comment_train.json --output-dir outputs/text_embedding --text-key comment --name-key name --max-length 512
```

Test comments:

```text
python scripts/extract_text_embeddings.py --split-json data/text_comments/video_comment_test.json --output-dir outputs/text_embedding --text-key comment --name-key name --max-length 512
```

## Step 2: Extract Video Features

```text
python scripts/extract_video_features.py --video-root data/raw_videos --split-json data/splits/train_data_new.json --output-dir outputs/video_feature --num-segments 48 --input-size 448
python scripts/extract_video_features.py --video-root data/raw_videos --split-json data/splits/test_data.json --output-dir outputs/video_feature --num-segments 48 --input-size 448
```

## Step 3: Extract 3D Keypoints from Videos

```text
git clone https://github.com/NationalGAILab/HoT.git ../HoT
python scripts/extract_video_keypoints_hot.py --hot-root ../HoT --video-root data/raw_videos --split-json data/splits/train_data_new.json --output-dir data/raw_keypoints --python /path/to/hot/env/bin/python --gpu 0
python scripts/extract_video_keypoints_hot.py --hot-root ../HoT --video-root data/raw_videos --split-json data/splits/test_data.json --output-dir data/raw_keypoints --python /path/to/hot/env/bin/python --gpu 0
```

## Step 4: Extract Keypoint Features

```text
python scripts/extract_keypoint_features.py --input-root data/raw_keypoints --split-json data/splits/train_data_new.json --output-dir outputs/keypoint_feature --segments 20
python scripts/extract_keypoint_features.py --input-root data/raw_keypoints --split-json data/splits/test_data.json --output-dir outputs/keypoint_feature --segments 20
```

## Step 5: Inspect One Sample

Use this to confirm the feature files are readable and have the expected shapes:

```text
python scripts/inspect_sample.py --name SAMPLE_NAME --video-root outputs/video_feature --keypoint-root outputs/keypoint_feature --embedding-root outputs/diffusion_embedding
```

## Step 6: Train Diffusion from Scratch

```text
python scripts/train_diffusion.py --train-json data/splits/train_data_new.json --text-root outputs/text_embedding --video-root outputs/video_feature --keypoint-root outputs/keypoint_feature --output-dir checkpoints/diffusion --device cuda:0 --schedule cosine --epochs 300 --batch-size 8 --lr 1e-4 --seed 3407 --save-every 10
```

## Step 7: Generate Diffusion Embeddings

Train split:

```text
python scripts/generate_diffusion_embeddings.py --split-json data/splits/train_data_new.json --video-root outputs/video_feature --keypoint-root outputs/keypoint_feature --output-root outputs/diffusion_embedding --diffusion-checkpoint checkpoints/diffusion/Diffusion.pt --reshape-keypoint-checkpoint checkpoints/diffusion/reshape_keypoint_module.pt --reshape-all-checkpoint checkpoints/diffusion/reshape_all_module.pt --device cuda:0 --schedule cosine
```

Test split:

```text
python scripts/generate_diffusion_embeddings.py --split-json data/splits/test_data.json --video-root outputs/video_feature --keypoint-root outputs/keypoint_feature --output-root outputs/diffusion_embedding --diffusion-checkpoint checkpoints/diffusion/Diffusion.pt --reshape-keypoint-checkpoint checkpoints/diffusion/reshape_keypoint_module.pt --reshape-all-checkpoint checkpoints/diffusion/reshape_all_module.pt --device cuda:0 --schedule cosine
```

## Step 8: Train Score Predictor

```text
python scripts/train_score.py --train-json data/splits/train_data_new.json --test-json data/splits/test_data.json --embedding-root outputs/diffusion_embedding --video-root outputs/video_feature --output-checkpoint checkpoints/score/predict_model.pt --device cuda:0 --epochs 160 --batch-size 8 --eval-batch-size 16 --lr 1e-4 --seed 3407
```

## Step 9: Predict Scores

```text
python scripts/predict_scores.py --split-json data/splits/test_data.json --embedding-root outputs/diffusion_embedding --video-root outputs/video_feature --predict-checkpoint checkpoints/score/predict_model.pt --output outputs/pred_scores.json --device cuda:0 --batch-size 8
```

## Step 10: Evaluate Predictions

```text
python scripts/evaluate_scores.py --pred-json outputs/pred_scores.json --pred-key pred_score --target-key score
```

The evaluation prints `rho`, `rl2`, `mse`, and `mae`.

## Step 11: MCTS Planning

First generate diffusion embeddings with `--save-steps`, because MCTS reads the cached denoising states for each diffusion step:

```text
python scripts/generate_diffusion_embeddings.py --split-json data/splits/test_data.json --video-root outputs/video_feature --keypoint-root outputs/keypoint_feature --output-root outputs/denoise_steps --diffusion-checkpoint checkpoints/diffusion/Diffusion.pt --reshape-keypoint-checkpoint checkpoints/diffusion/reshape_keypoint_module.pt --reshape-all-checkpoint checkpoints/diffusion/reshape_all_module.pt --device cuda:0 --schedule cosine --save-steps
```

Then run MCTS planning. The selected MCTS plan for each sample is saved to the JSON file specified by `--output`.

```text
python scripts/plan_mcts.py --split-json data/splits/test_data.json --pred-score-json outputs/pred_scores.json --video-root outputs/video_feature --keypoint-root outputs/keypoint_feature --denoise-root outputs/denoise_steps --output outputs/mcts_plans.json --diffusion-checkpoint checkpoints/diffusion/Diffusion.pt --reshape-keypoint-checkpoint checkpoints/diffusion/reshape_keypoint_module.pt --reshape-all-checkpoint checkpoints/diffusion/reshape_all_module.pt --predict-checkpoint checkpoints/score/predict_model.pt --device cuda:0 --schedule cosine --num-simulations 150
```

## Useful CLI Help

Each script exposes argparse help:

```text
python scripts/extract_text_embeddings.py --help
python scripts/extract_video_keypoints_hot.py --help
python scripts/train_diffusion.py --help
python scripts/generate_diffusion_embeddings.py --help
python scripts/train_score.py --help
python scripts/predict_scores.py --help
python scripts/plan_mcts.py --help
```
