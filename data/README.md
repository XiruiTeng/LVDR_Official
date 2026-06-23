# Data Layout

Put local data under this directory before running the pipeline.

Required split files:

- `data/splits/train_data_new.json`
- `data/splits/test_data.json`

Required text-comment files for text embedding extraction:

- `data/text_comments/video_comment_train.json`
- `data/text_comments/video_comment_test.json`

Raw feature inputs, if extracting from raw data:

- `data/raw_videos/<name>.mp4`
- `data/raw_keypoints/<name>.npz`
- or `data/raw_keypoints/<name>/output_3D/*.npz`

Video extraction writes `outputs/video_feature/<name>.pt`.

Video-to-keypoint extraction with HoT writes
`data/raw_keypoints/<name>/output_3D/output_keypoints_3d.npz`.

Keypoint extraction writes `outputs/keypoint_feature/<name>.pt`. Raw `.npz` keypoints should
contain a `reconstruction` array by default, or you can pass `--npz-key` to select another array.
The accepted raw keypoint shapes are `(frames, joints, dims)` for a full sequence or `(joints, dims)`
for per-frame files inside `output_3D/`.

Processed feature inputs, if reusing cached features:

- `data/processed_features/video_feature/<name>.pt`
- `data/processed_features/keypoint_feature/<name>.pt`

Generated outputs and checkpoints are intentionally not tracked by git.
