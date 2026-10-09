"""Models that look at one turbine's own history at a time (no spatial information).

Every model takes (batch, turbines, steps, features) and returns (batch, turbines); the turbine axis
is just extra batch rows here, so one set of weights is shared by all turbines.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMModel(nn.Module):
    def __init__(self, in_features: int = 5, hidden: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(in_features, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, steps, f = x.shape
        out, _ = self.lstm(x.reshape(b * n, steps, f))
        return self.fc(out[:, -1]).reshape(b, n)


class CausalConv(nn.Module):
    """1-D convolution padded on the left only, so an output never depends on later steps."""

    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, kernel, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.pad, 0)))


class TCNModel(nn.Module):
    """Three causal convolutions with dilations 1, 2, 4: the receptive field is 15 steps, a full window."""

    def __init__(self, in_features: int = 5, channels: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            CausalConv(in_features, channels, 3, 1), nn.ReLU(),
            CausalConv(channels, channels, 3, 2), nn.ReLU(),
            CausalConv(channels, channels, 3, 4), nn.ReLU(),
        )
        self.fc = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, steps, f = x.shape
        out = self.network(x.reshape(b * n, steps, f).permute(0, 2, 1))  # (rows, channels, steps)
        return self.fc(out[:, :, -1]).reshape(b, n)


class TransformerBlock(nn.Module):
    """One self-attention block over the time steps; each step is the readings of all turbines side by side."""

    def __init__(self, in_dim: int, out_nodes: int, steps: int = 12, d_model: int = 128, heads: int = 4):
        super().__init__()
        self.embedding = nn.Linear(in_dim, d_model)
        self.position = nn.Parameter(torch.zeros(1, steps, d_model))
        self.attention = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, out_nodes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # (batch, steps, in_dim) -> (batch, out_nodes)
        x = self.embedding(tokens) + self.position
        attn, _ = self.attention(x, x, x)
        x = self.norm1(x + attn)
        x = self.norm2(x + self.ff(x))
        return self.output(x[:, -1])


class TransformerModel(nn.Module):
    """Temporal self-attention over the readings of all turbines together (a single block, no graph)."""

    def __init__(self, n_nodes: int, in_features: int = 5, steps: int = 12):
        super().__init__()
        self.block = TransformerBlock(n_nodes * in_features, n_nodes, steps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, steps, f = x.shape
        return self.block(x.permute(0, 2, 1, 3).reshape(b, steps, n * f))
