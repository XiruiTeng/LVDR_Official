# [ECCV 2026] [Latent Visual Diffusion Reasoning with Monte Carlo Tree Search](https://arxiv.org/abs/2606.27988)

The official implementation of _Latent Visual Diffusion Reasoning with Monte Carlo Tree Search_

**[Xirui Teng](https://xiruiteng.github.io), [Nan Xi*](https://southnx.github.io/), [Junsong Yuan](https://cse.buffalo.edu/~jsyuan/index.html)**

## Before Training

This stage prepares the environment, input data, text descriptions, and visual/keypoint features
needed by the training pipeline.

### Environment

Create and activate a conda environment first:

  ```bash
  conda create -n lvdr_official python=3.10 -y
  conda activate lvdr_official
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  ```

### Prepare Data

Prepare the required files under the `data/` directory. Each subdirectory is used as follows:

- `data/splits/`: stores the train/test split JSON files, such as `train_data.json` and `test_data.json`.
- `data/text_comments/`: stores video-level text comments or descriptions used for text embedding extraction.
- `data/raw_videos/`: stores raw video files used for video feature extraction and video-based comment generation.
- `data/raw_keypoints/`: stores raw keypoint files used for keypoint feature extraction.
- `data/processed_features/`: stores optional precomputed visual and keypoint features if feature extraction is skipped.

### Generate Video Comment

Generate video-level comments from raw videos with the Qwen2.5-VL entry point. Replace
`<prompt_preset>` according to the dataset being processed.

```text
python scripts/generate_qwen_video_comments.py --video-root data/raw_videos --output-json data/text_comments/video_comment_train.json --split-json data/splits/train_data.json --prompt-preset <prompt_preset> --output-text-key comment --fps 1.0 --attn-implementation flash_attention_2
python scripts/generate_qwen_video_comments.py --video-root data/raw_videos --output-json data/text_comments/video_comment_test.json --split-json data/splits/test_data.json --prompt-preset <prompt_preset> --output-text-key comment --fps 1.0 --attn-implementation flash_attention_2
```

### Extract Text Embeddings

Convert the generated video comments into text embeddings. The train and test comment files should
be processed separately.

Train comments:

```text
python scripts/extract_text_embeddings.py --split-json data/text_comments/video_comment_train.json --output-dir outputs/text_embedding --text-key comment --name-key name --max-length 512
```

Test comments:

```text
python scripts/extract_text_embeddings.py --split-json data/text_comments/video_comment_test.json --output-dir outputs/text_embedding --text-key comment --name-key name --max-length 512
```

### Extract Video Features

Extract visual features from the raw videos for both the training and test splits. The extracted
features are saved under `outputs/video_feature`.

```text
python scripts/extract_video_features.py --video-root data/raw_videos --split-json data/splits/train_data.json --output-dir outputs/video_feature --num-segments 48 --input-size 448
python scripts/extract_video_features.py --video-root data/raw_videos --split-json data/splits/test_data.json --output-dir outputs/video_feature --num-segments 48 --input-size 448
```

### Extract 3D Keypoints from Videos

First clone [HoT](https://github.com/NationalGAILab/HoT):

```text
git clone https://github.com/NationalGAILab/HoT.git ../HoT
```

Then set up a separate HoT environment following the official HoT instructions.

Place the HoT repository next to this project directory, so the expected path is:

```text
../HoT
```

After HoT is installed and its environment is ready, extract 3D keypoints with:

```text
python scripts/extract_video_keypoints_hot.py --hot-root ../HoT --video-root data/raw_videos --split-json data/splits/train_data.json --output-dir data/raw_keypoints --python /path/to/hot/env/bin/python --gpu 0
python scripts/extract_video_keypoints_hot.py --hot-root ../HoT --video-root data/raw_videos --split-json data/splits/test_data.json --output-dir data/raw_keypoints --python /path/to/hot/env/bin/python --gpu 0
```

### Extract Keypoint Features

Convert the raw 3D keypoints into fixed-length keypoint features used by the diffusion model.
The extracted features are saved under `outputs/keypoint_feature`.

```text
python scripts/extract_keypoint_features.py --input-root data/raw_keypoints --split-json data/splits/train_data.json --output-dir outputs/keypoint_feature --segments 20
python scripts/extract_keypoint_features.py --input-root data/raw_keypoints --split-json data/splits/test_data.json --output-dir outputs/keypoint_feature --segments 20
```

## Training Pipeline

After all required features are prepared, train the diffusion model and the score predictor.

### Train Diffusion from Scratch

Train the latent diffusion model using text embeddings, video features, and keypoint features from
the training split.

```text
python scripts/train_diffusion.py --train-json data/splits/train_data.json --text-root outputs/text_embedding --video-root outputs/video_feature --keypoint-root outputs/keypoint_feature --output-dir checkpoints/diffusion --device cuda:0 --schedule cosine --epochs 300 --batch-size 8 --lr 1e-4 --save-every 10
```

### Generate Diffusion Embeddings

Use the trained diffusion checkpoint to generate latent diffusion embeddings. These embeddings are
used by the score predictor and should be generated for both train and test splits.

Train split:

```text
python scripts/generate_diffusion_embeddings.py --split-json data/splits/train_data.json --video-root outputs/video_feature --keypoint-root outputs/keypoint_feature --output-root outputs/diffusion_embedding --diffusion-checkpoint checkpoints/diffusion/Diffusion.pt --reshape-keypoint-checkpoint checkpoints/diffusion/reshape_keypoint_module.pt --reshape-all-checkpoint checkpoints/diffusion/reshape_all_module.pt --device cuda:0 --schedule cosine
```

Test split:

```text
python scripts/generate_diffusion_embeddings.py --split-json data/splits/test_data.json --video-root outputs/video_feature --keypoint-root outputs/keypoint_feature --output-root outputs/diffusion_embedding --diffusion-checkpoint checkpoints/diffusion/Diffusion.pt --reshape-keypoint-checkpoint checkpoints/diffusion/reshape_keypoint_module.pt --reshape-all-checkpoint checkpoints/diffusion/reshape_all_module.pt --device cuda:0 --schedule cosine
```

### Train Score Predictor

Train the score predictor using the generated diffusion embeddings and video features.

```text
python scripts/train_score.py --train-json data/splits/train_data.json --test-json data/splits/test_data.json --embedding-root outputs/diffusion_embedding --video-root outputs/video_feature --output-checkpoint checkpoints/score/predict_model.pt --device cuda:0 --epochs 160 --batch-size 8 --eval-batch-size 16 --lr 1e-4
```

## Inference Pipeline

Use the trained checkpoints to predict test scores and run MCTS planning.

### Predict Scores

Predict scores for the test split with the trained score predictor.

```text
python scripts/predict_scores.py --split-json data/splits/test_data.json --embedding-root outputs/diffusion_embedding --video-root outputs/video_feature --predict-checkpoint checkpoints/score/predict_model.pt --output outputs/pred_scores.json --device cuda:0 --batch-size 8
```

### MCTS Planning

First generate diffusion embeddings with `--save-steps`, because MCTS reads the cached denoising states for each diffusion step:

```text
python scripts/generate_diffusion_embeddings.py --split-json data/splits/test_data.json --video-root outputs/video_feature --keypoint-root outputs/keypoint_feature --output-root outputs/denoise_steps --diffusion-checkpoint checkpoints/diffusion/Diffusion.pt --reshape-keypoint-checkpoint checkpoints/diffusion/reshape_keypoint_module.pt --reshape-all-checkpoint checkpoints/diffusion/reshape_all_module.pt --device cuda:0 --schedule cosine --save-steps
```

Then run MCTS planning. The selected MCTS plan for each sample is saved to the JSON file specified by `--output`.

```text
python scripts/plan_mcts.py --split-json data/splits/test_data.json --pred-score-json outputs/pred_scores.json --video-root outputs/video_feature --keypoint-root outputs/keypoint_feature --denoise-root outputs/denoise_steps --output outputs/mcts_plans.json --diffusion-checkpoint checkpoints/diffusion/Diffusion.pt --reshape-keypoint-checkpoint checkpoints/diffusion/reshape_keypoint_module.pt --reshape-all-checkpoint checkpoints/diffusion/reshape_all_module.pt --predict-checkpoint checkpoints/score/predict_model.pt --device cuda:0 --schedule cosine --num-simulations 150
```
