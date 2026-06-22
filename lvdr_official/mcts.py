from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .data import load_feature_pair
from .diffusion import NoiseSchedule, load_diffusion_bundle
from .models import DiTConv, PredictModel, ReshapeAll, ReshapeKeypoint
from .score import load_predict_model
from .utils import ensure_dir, limit_items, load_json, load_tensor, save_json


class TreeNode:
    def __init__(self, parent: "TreeNode | None", joint_id: int | None = None, num_total_joints: int = 4) -> None:
        self.parent = parent
        self.children: list[TreeNode] = []
        self._num_total_joints = num_total_joints
        self.selected_joint: list[int] = []
        self.depth = 0
        if parent:
            self.selected_joint.extend(parent.selected_joint)
            self.depth = parent.depth + 1
        if joint_id is not None:
            self.selected_joint.append(joint_id)
        self.visit_count = 0
        self.value_sum = 0.0

    @property
    def value(self) -> float:
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count

    def get_uct_value(self, exploration_weight: float = math.sqrt(2.0)) -> float:
        if self.visit_count == 0:
            return float("inf")
        parent_visits = self.parent.visit_count if self.parent else self.visit_count
        exploration = exploration_weight * math.sqrt(
            math.log(parent_visits + 1e-6) / (self.visit_count + 1e-6)
        )
        return self.value + exploration

    def select_best_child(self) -> "TreeNode":
        return max(self.children, key=lambda child: child.get_uct_value())

    def expand(self) -> None:
        for joint_id in range(self._num_total_joints):
            self.children.append(TreeNode(self, joint_id, self._num_total_joints))

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def backpropagate(self, reward: float) -> None:
        node: TreeNode | None = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += reward
            node = node.parent


@torch.no_grad()
def construct_mcts_prompt(
    joint_combination: list[int],
    step_index: torch.Tensor,
    video_feature: torch.Tensor,
    keypoint_feature: torch.Tensor,
    reshape_keypoint: ReshapeKeypoint,
    reshape_all: ReshapeAll,
    new_keypoint_feature: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = video_feature.device
    t_indices = torch.arange(48, device=device).unsqueeze(0)
    index = int(step_index.item())

    this_elbow = keypoint_feature[:, index - 1 : index, 0:10]
    this_shoulder = keypoint_feature[:, index - 1 : index, 10:20]
    this_hip = keypoint_feature[:, index - 1 : index, 20:30]
    this_knee = keypoint_feature[:, index - 1 : index, 30:40]
    all_keypoints = [this_elbow, this_shoulder, this_hip, this_knee]

    selected_keypoint = torch.cat([all_keypoints[i] for i in joint_combination], dim=2)
    new_keypoint_feature[:, index - 1 : index, :] = selected_keypoint

    mask = (t_indices < step_index.unsqueeze(1)).unsqueeze(-1).float()
    num_valid_frames = mask.sum(dim=1).clamp(min=1)
    avg_video = (video_feature * mask).sum(dim=1) / num_valid_frames
    avg_keypoint = (new_keypoint_feature * mask).sum(dim=1) / num_valid_frames

    x_prompt_vision = avg_video.unsqueeze(1)
    x_prompt_action = avg_keypoint.unsqueeze(1)
    prompts = [
        reshape_keypoint(x_prompt_action[:, :, 0:10]),
        reshape_keypoint(x_prompt_action[:, :, 10:20]),
        reshape_keypoint(x_prompt_action[:, :, 20:30]),
        reshape_keypoint(x_prompt_action[:, :, 30:40]),
    ]
    all_prompts_tensor = torch.cat(prompts, dim=1)
    x_tokens = reshape_all(torch.cat([x_prompt_vision] + prompts, dim=1))
    return x_tokens, new_keypoint_feature, all_prompts_tensor


@torch.no_grad()
def evaluate_plan_for_step(
    denoise_embedding: torch.Tensor,
    plan: list[int],
    step_index: torch.Tensor,
    video_feature: torch.Tensor,
    keypoint_feature: torch.Tensor,
    reshape_keypoint: ReshapeKeypoint,
    reshape_all: ReshapeAll,
    base_new_keypoint_feature: torch.Tensor,
) -> float:
    tmp_new_keypoint = base_new_keypoint_feature.clone()
    _, _, all_prompts_tensor = construct_mcts_prompt(
        plan,
        step_index,
        video_feature,
        keypoint_feature,
        reshape_keypoint,
        reshape_all,
        tmp_new_keypoint,
    )
    prompt_vec = all_prompts_tensor.mean(dim=1)
    sim = F.cosine_similarity(denoise_embedding, prompt_vec, dim=-1)
    return float(sim.mean().item())


@torch.no_grad()
def mcts_for_single_step(
    denoise_embedding: torch.Tensor,
    step_index: torch.Tensor,
    video_feature: torch.Tensor,
    keypoint_feature: torch.Tensor,
    reshape_keypoint: ReshapeKeypoint,
    reshape_all: ReshapeAll,
    base_new_keypoint_feature: torch.Tensor,
    num_simulations: int = 150,
    num_joints: int = 4,
    max_depth: int = 4,
) -> list[int]:
    root = TreeNode(parent=None, num_total_joints=num_joints)
    for _ in range(num_simulations):
        node = root
        while not node.is_leaf():
            node = node.select_best_child()
        if node.depth < max_depth:
            node.expand()

        current_plan = node.selected_joint
        remaining_depth = max_depth - len(current_plan)
        rollout = [random.randint(0, num_joints - 1) for _ in range(remaining_depth)]
        reward = evaluate_plan_for_step(
            denoise_embedding,
            current_plan + rollout,
            step_index,
            video_feature,
            keypoint_feature,
            reshape_keypoint,
            reshape_all,
            base_new_keypoint_feature,
        )
        node.backpropagate(reward)

    best_plan: list[int] = []
    node = root
    while len(best_plan) < max_depth:
        if not node.children:
            best_plan.extend([random.randint(0, num_joints - 1) for _ in range(max_depth - len(best_plan))])
            break
        node = max(node.children, key=lambda child: child.visit_count)
        best_plan = node.selected_joint
    return best_plan


def load_step_embedding(denoise_root: str | Path, name: str, step: int, device: torch.device) -> torch.Tensor:
    denoise_root = Path(denoise_root)
    candidates = [
        denoise_root / f"{name}_{step}.pt",
        denoise_root / name / f"z_step_{step}.pt",
        denoise_root / name / f"z_step_{step - 1}.pt",
    ]
    for path in candidates:
        if path.exists():
            return load_tensor(path, map_location=device).to(device=device, dtype=torch.float32)
    raise FileNotFoundError(f"No cached denoising embedding found for {name} step {step}.")


@torch.no_grad()
def sample_step_mcts(
    denoise_root: str | Path,
    name: str,
    diffusion_model: DiTConv,
    predict_model: PredictModel,
    reshape_keypoint: ReshapeKeypoint,
    reshape_all: ReshapeAll,
    video_feature: torch.Tensor,
    keypoint_feature: torch.Tensor,
    schedule: NoiseSchedule,
    num_simulations: int = 150,
) -> tuple[torch.Tensor, list[list[int]]]:
    del predict_model
    device = video_feature.device
    batch = video_feature.shape[0]
    z_t = torch.randn(
        batch,
        diffusion_model.T,
        diffusion_model.proj_out.out_features,
        device=device,
    )
    new_keypoint_feature = keypoint_feature.clone()
    best_plan_history: list[list[int]] = []

    for t in range(schedule.num_steps, 0, -1):
        step_index = torch.full((batch,), t, device=device, dtype=torch.long)
        denoise_embedding = load_step_embedding(denoise_root, name, t, device)
        best_plan = mcts_for_single_step(
            denoise_embedding,
            step_index,
            video_feature,
            keypoint_feature,
            reshape_keypoint,
            reshape_all,
            new_keypoint_feature,
            num_simulations=num_simulations,
        )
        best_plan_history.append(best_plan)
        x_tokens, new_keypoint_feature, _ = construct_mcts_prompt(
            best_plan,
            step_index,
            video_feature,
            keypoint_feature,
            reshape_keypoint,
            reshape_all,
            new_keypoint_feature,
        )

        a_bar_t = schedule.alpha_bar(step_index).view(batch, 1, 1).to(device)
        a_bar_tm1 = schedule.alpha_bar(step_index - 1).view(batch, 1, 1).to(device)
        eps_hat = diffusion_model(z_t, step_index, x_tokens, use_causal_self=True)
        alpha_t = a_bar_t / a_bar_tm1
        beta_t = 1.0 - alpha_t
        z_t = (
            z_t
            - (beta_t / torch.sqrt(1.0 - a_bar_t + 1e-8)) * eps_hat
        ) / torch.sqrt(alpha_t + 1e-8)

    return z_t, best_plan_history


def plan_dataset(
    split_json: str | Path,
    pred_score_json: str | Path,
    video_root: str | Path,
    keypoint_root: str | Path,
    denoise_root: str | Path,
    output_json: str | Path,
    diffusion_checkpoint: str | Path,
    reshape_keypoint_checkpoint: str | Path,
    reshape_all_checkpoint: str | Path,
    predict_checkpoint: str | Path,
    device: torch.device,
    schedule: NoiseSchedule,
    max_samples: int | None = None,
    num_simulations: int = 150,
) -> list[dict[str, Any]]:
    diffusion_model, reshape_keypoint, reshape_all = load_diffusion_bundle(
        diffusion_checkpoint,
        reshape_keypoint_checkpoint,
        reshape_all_checkpoint,
        device,
    )
    predict_model = load_predict_model(predict_checkpoint, device)
    split_items = limit_items(load_json(split_json), max_samples)
    pred_by_name = {item["name"]: item for item in load_json(pred_score_json)}
    results: list[dict[str, Any]] = []

    for item in tqdm(split_items, desc="planning with mcts"):
        name = item["name"]
        video, keypoint = load_feature_pair(name, video_root, keypoint_root, device)
        _, plan = sample_step_mcts(
            denoise_root,
            name,
            diffusion_model,
            predict_model,
            reshape_keypoint,
            reshape_all,
            video,
            keypoint,
            schedule,
            num_simulations=num_simulations,
        )
        score_item = pred_by_name.get(name, item)
        results.append(
            {
                "name": name,
                "score": float(score_item.get("pred_score", score_item.get("score", 0.0))),
                "plan": plan,
            }
        )

    ensure_dir(Path(output_json).parent)
    save_json(results, output_json)
    return results
