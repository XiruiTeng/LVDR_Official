from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import DiffusionTrainingDataset, load_feature_pair
from .models import DiTConv, ReshapeAll, ReshapeKeypoint
from .utils import ensure_dir, limit_items, load_json, load_state_dict, seed_everything


@dataclass(frozen=True)
class NoiseSchedule:
    kind: str = "linear"
    num_steps: int = 48
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    def alpha_bar(self, steps: torch.Tensor) -> torch.Tensor:
        steps = steps.to(dtype=torch.long)
        if self.kind == "linear":
            alpha_bars = torch.cumprod(
                1.0
                - torch.linspace(
                    self.beta_start,
                    self.beta_end,
                    self.num_steps,
                    dtype=torch.float32,
                    device=steps.device,
                ),
                dim=0,
            )
            idx = (steps - 1).clamp(0, self.num_steps - 1)
            values = alpha_bars.gather(0, idx)
            return torch.where(steps <= 0, torch.ones_like(values), values)
        if self.kind == "cosine":
            f = (steps.float() + 0.5) / self.num_steps
            values = torch.cos((f + 0.0008) / 1.0008 * math.pi / 2) ** 2
            return torch.where(steps <= 0, torch.ones_like(values), values)
        raise ValueError(f"Unknown noise schedule: {self.kind}")


def add_noise_eps(
    z0: torch.Tensor,
    step_index: torch.Tensor,
    schedule: NoiseSchedule,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = z0.shape[0]
    a_bar = schedule.alpha_bar(step_index).view(batch, 1, 1).to(z0.device)
    eps = torch.randn_like(z0)
    z_t = torch.sqrt(a_bar) * z0 + torch.sqrt(1.0 - a_bar) * eps
    return z_t, eps, a_bar


def split_raw_keypoints(keypoint: torch.Tensor) -> tuple[torch.Tensor, ...]:
    elbow = torch.cat([keypoint[:, :, 25:30], keypoint[:, :, 35:40]], dim=2)
    shoulder = torch.cat([keypoint[:, :, 20:25], keypoint[:, :, 30:35]], dim=2)
    hip = torch.cat([keypoint[:, :, 10:15], keypoint[:, :, 0:5]], dim=2)
    knee = torch.cat([keypoint[:, :, 15:20], keypoint[:, :, 5:10]], dim=2)
    return elbow, shoulder, hip, knee


def condition_tokens_for_step(
    video_feature: torch.Tensor,
    keypoint_feature: torch.Tensor,
    step_index: torch.Tensor,
    reshape_keypoint: ReshapeKeypoint,
    reshape_all: ReshapeAll,
) -> torch.Tensor:
    x_raw = torch.cat([video_feature, keypoint_feature], dim=-1)
    batch, tx_raw, channels = x_raw.shape
    t_idx = (step_index - 1).clamp(min=0, max=tx_raw - 1)
    gather_idx = t_idx.view(batch, 1, 1).expand(-1, 1, channels)
    x_prompt = torch.gather(x_raw, dim=1, index=gather_idx)
    x_prompt_vision = x_prompt[:, :, :4096]
    x_prompt_action = x_prompt[:, :, -40:]
    prompts = [reshape_keypoint(part) for part in split_raw_keypoints(x_prompt_action)]
    condition_prompt = torch.cat([x_prompt_vision] + prompts, dim=1)
    return reshape_all(condition_prompt)


def load_diffusion_bundle(
    diffusion_checkpoint: str | Path,
    reshape_keypoint_checkpoint: str | Path,
    reshape_all_checkpoint: str | Path,
    device: torch.device,
    use_causal_cross: bool = False,
) -> tuple[DiTConv, ReshapeKeypoint, ReshapeAll]:
    diffusion_model = DiTConv(use_causal_cross=use_causal_cross)
    reshape_keypoint = ReshapeKeypoint()
    reshape_all = ReshapeAll()
    diffusion_model.load_state_dict(load_state_dict(diffusion_checkpoint))
    reshape_keypoint.load_state_dict(load_state_dict(reshape_keypoint_checkpoint))
    reshape_all.load_state_dict(load_state_dict(reshape_all_checkpoint))
    return (
        diffusion_model.to(device).eval(),
        reshape_keypoint.to(device).eval(),
        reshape_all.to(device).eval(),
    )


@torch.no_grad()
def sample_text_embedding(
    diffusion_model: DiTConv,
    reshape_keypoint: ReshapeKeypoint,
    reshape_all: ReshapeAll,
    video_feature: torch.Tensor,
    keypoint_feature: torch.Tensor,
    schedule: NoiseSchedule,
    eta: float = 0.0,
    use_causal_self: bool = True,
    save_steps_dir: str | Path | None = None,
    name: str | None = None,
) -> torch.Tensor:
    device = video_feature.device
    batch = video_feature.shape[0]
    z_t = torch.randn(
        batch,
        diffusion_model.T,
        diffusion_model.proj_out.out_features,
        device=device,
    )
    if save_steps_dir is not None:
        if name is None:
            raise ValueError("name must be provided when save_steps_dir is set.")
        ensure_dir(save_steps_dir)

    for t in range(schedule.num_steps, 0, -1):
        step_index = torch.full((batch,), t, device=device, dtype=torch.long)
        x_tokens = condition_tokens_for_step(
            video_feature,
            keypoint_feature,
            step_index,
            reshape_keypoint,
            reshape_all,
        )
        a_bar_t = schedule.alpha_bar(step_index).view(batch, 1, 1).to(device)
        a_bar_tm1 = schedule.alpha_bar(step_index - 1).view(batch, 1, 1).to(device)

        if t > 1:
            eps_hat = diffusion_model(z_t, step_index, x_tokens, use_causal_self=use_causal_self)
        else:
            eps_hat = (z_t - torch.sqrt(a_bar_t) * x_tokens) / torch.sqrt(1.0 - a_bar_t + 1e-8)

        alpha_t = a_bar_t / a_bar_tm1
        beta_t = 1.0 - alpha_t
        tilde_beta_t = (1.0 - a_bar_tm1) / (1.0 - a_bar_t) * beta_t
        sigma_t = torch.sqrt(torch.clamp(tilde_beta_t, min=0.0)) * (eta if t > 1 else 0.0)
        noise = torch.randn_like(z_t) if (t > 1 and eta > 0) else torch.zeros_like(z_t)
        z_t = (
            z_t
            - (beta_t / torch.sqrt(1.0 - a_bar_t + 1e-8)) * eps_hat
        ) / torch.sqrt(alpha_t + 1e-8) + sigma_t * noise

        if save_steps_dir is not None:
            torch.save(z_t.detach().cpu(), Path(save_steps_dir) / f"{name}_{t}.pt")

    return z_t


def generate_embeddings(
    split_json: str | Path,
    video_root: str | Path,
    keypoint_root: str | Path,
    output_root: str | Path,
    diffusion_model: DiTConv,
    reshape_keypoint: ReshapeKeypoint,
    reshape_all: ReshapeAll,
    schedule: NoiseSchedule,
    device: torch.device,
    max_samples: int | None = None,
    save_steps: bool = False,
) -> list[Path]:
    output_root = ensure_dir(output_root)
    items = limit_items(load_json(split_json), max_samples)
    written: list[Path] = []
    for item in tqdm(items, desc="generating embeddings"):
        name = item["name"]
        video, keypoint = load_feature_pair(name, video_root, keypoint_root, device)
        z0 = sample_text_embedding(
            diffusion_model,
            reshape_keypoint,
            reshape_all,
            video,
            keypoint,
            schedule,
            save_steps_dir=output_root if save_steps else None,
            name=name,
        )
        output_path = output_root / f"{name}.pt"
        torch.save(z0.cpu(), output_path)
        written.append(output_path)
    return written


def train_diffusion(
    train_json: str | Path,
    text_root: str | Path,
    video_root: str | Path,
    keypoint_root: str | Path,
    output_dir: str | Path,
    device: torch.device,
    schedule: NoiseSchedule,
    epochs: int = 300,
    batch_size: int = 8,
    lr: float = 1e-4,
    seed: int = 3407,
    max_samples: int | None = None,
    save_every: int = 10,
) -> None:
    seed_everything(seed)
    output_dir = ensure_dir(output_dir)
    dataset = DiffusionTrainingDataset(train_json, text_root, video_root, keypoint_root)
    if max_samples is not None and max_samples > 0:
        dataset.items = limit_items(dataset.items, max_samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    diffusion_model = DiTConv().to(device)
    reshape_keypoint = ReshapeKeypoint().to(device)
    reshape_all = ReshapeAll().to(device)
    optimizers = [
        torch.optim.Adam(diffusion_model.parameters(), lr=lr),
        torch.optim.Adam(reshape_keypoint.parameters(), lr=lr),
        torch.optim.Adam(reshape_all.parameters(), lr=lr),
    ]

    for epoch in range(epochs):
        running_loss = 0.0
        for batch in loader:
            z0 = batch["text_embedding"].to(device=device, dtype=torch.float32)
            video = batch["video_feature"].to(device=device, dtype=torch.float32)
            keypoint = batch["keypoint_feature"].to(device=device, dtype=torch.float32)
            step_index = torch.randint(1, schedule.num_steps + 1, (z0.shape[0],), device=device)
            z_s, eps, _ = add_noise_eps(z0, step_index, schedule)
            x_tokens = condition_tokens_for_step(
                video,
                keypoint,
                step_index,
                reshape_keypoint,
                reshape_all,
            )
            eps_hat = diffusion_model(z_s, step_index, x_tokens, use_causal_self=True)
            first_mask = step_index == 1
            if first_mask.any():
                a_bar = schedule.alpha_bar(step_index).view(z0.shape[0], 1, 1).to(device)
                eps_first = (z_s - torch.sqrt(a_bar) * x_tokens) / torch.sqrt(1.0 - a_bar + 1e-8)
                eps_hat = eps_hat.clone()
                eps_hat[first_mask] = eps_first[first_mask]

            loss = F.mse_loss(eps_hat, eps)
            for optimizer in optimizers:
                optimizer.zero_grad()
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()
            running_loss += float(loss.item())

        if epoch % save_every == 0 or epoch + 1 == epochs:
            print(f"epoch={epoch} loss={running_loss / max(len(loader), 1):.6f}")
            torch.save(diffusion_model.state_dict(), output_dir / "Diffusion.pt")
            torch.save(reshape_keypoint.state_dict(), output_dir / "reshape_keypoint_module.pt")
            torch.save(reshape_all.state_dict(), output_dir / "reshape_all_module.pt")
