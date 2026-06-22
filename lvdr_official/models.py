from __future__ import annotations

import torch
import torch.nn as nn


class SinRoPE(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe[None])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class DiffusionStepEmbedding(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, steps: torch.Tensor, d_model: int) -> torch.Tensor:
        half = d_model // 2
        freqs = torch.exp(
            -torch.arange(half, device=steps.device)
            * (torch.log(torch.tensor(10000.0, device=steps.device)) / half)
        )
        emb = steps.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.proj(emb)


class DiTBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int = 8, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ln3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(d_model * mlp_ratio), d_model),
        )

    def forward(
        self,
        z: torch.Tensor,
        hx: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        self_attn_mask: torch.Tensor | None = None,
        cross_attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z_norm = self.ln1(z)
        z = z + self.self_attn(z_norm, z_norm, z_norm, attn_mask=self_attn_mask)[0]
        z = z + self.cross_attn(
            self.ln2(z),
            hx,
            hx,
            key_padding_mask=key_padding_mask,
            attn_mask=cross_attn_mask,
        )[0]
        z = z + self.ffn(self.ln3(z))
        return z


class DiTConv(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        layers: int = 12,
        nhead: int = 16,
        T: int = 512,
        Tx: int = 512,
        out_dim_z: int = 4096,
        out_dim_x: int = 4096,
        v_param: bool = True,
        use_causal_cross: bool = False,
    ) -> None:
        super().__init__()
        self.T, self.Tx, self.d = T, Tx, d_model
        self.proj_in = nn.Linear(out_dim_z, d_model)
        self.proj_x = nn.Linear(out_dim_x, d_model)
        self.proj_out = nn.Linear(d_model, out_dim_z)
        self.pe_z = SinRoPE(d_model, T)
        self.pe_x = SinRoPE(d_model, Tx)
        self.t_embed = DiffusionStepEmbedding(d_model)
        self.blocks = nn.ModuleList([DiTBlock(d_model, nhead=nhead) for _ in range(layers)])
        self.ln_out = nn.LayerNorm(d_model)
        self.v_param = v_param
        self.use_causal_cross = use_causal_cross

        cross_mask = torch.ones(T, Tx, dtype=torch.bool)
        for t in range(T):
            cross_mask[t, : min(t, Tx - 1) + 1] = False
        self.register_buffer("attn_mask_x", cross_mask)

        self_mask = torch.ones(T, T, dtype=torch.bool)
        self_mask = torch.triu(self_mask, diagonal=1)
        self.register_buffer("self_attn_mask", self_mask)

    def forward(
        self,
        z_s: torch.Tensor,
        steps: torch.Tensor,
        x_tokens: torch.Tensor,
        use_causal_self: bool = True,
    ) -> torch.Tensor:
        z = self.pe_z(self.proj_in(z_s))
        x = self.pe_x(self.proj_x(x_tokens))
        z = z + self.t_embed(steps, self.d).unsqueeze(1)
        self_mask = self.self_attn_mask if use_causal_self else None
        cross_mask = self.attn_mask_x if self.use_causal_cross else None
        for block in self.blocks:
            z = block(z, x, self_attn_mask=self_mask, cross_attn_mask=cross_mask)
        return self.proj_out(self.ln_out(z))


class ReshapeKeypoint(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.reshape_module = nn.Sequential(
            nn.Linear(10, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4096),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps = x.shape[0], x.shape[1]
        x = x.reshape(batch, -1)
        x = self.reshape_module(x)
        return x.reshape(batch, steps, -1)


class ReshapeAll(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.reshape_module = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
        )

    def forward(self, all_feature: torch.Tensor) -> torch.Tensor:
        x = all_feature.transpose(1, 2)
        x = self.reshape_module(x)
        return x.transpose(1, 2)


class PredictModel(nn.Module):
    def __init__(self, input_shape: int = 48 * 4096) -> None:
        super().__init__()
        self.reshape = nn.Conv1d(560, 48, kernel_size=1)
        self.predict_module = nn.Sequential(
            nn.Linear(input_shape, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, embedding: torch.Tensor, visual: torch.Tensor) -> torch.Tensor:
        x = torch.cat([embedding, visual], dim=1)
        x = self.reshape(x)
        x = torch.flatten(x, 1)
        pred_score = self.predict_module(x)
        return 1 + 9 * torch.sigmoid(pred_score)
