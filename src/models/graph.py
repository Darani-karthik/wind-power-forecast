"""Models that also look at the neighbouring turbines through the k-nearest-neighbour graph.

Same interface as the temporal models: (batch, turbines, steps, features) -> (batch, turbines). All
`batch * steps` snapshots go through the graph layers as one big disjoint graph instead of a Python loop.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv

from .temporal import TransformerBlock


def batch_edges(edge_index: torch.Tensor, n_nodes: int, n_graphs: int) -> torch.Tensor:
    """Repeat the graph `n_graphs` times side by side, shifting the node ids of each copy."""
    shift = torch.arange(n_graphs, device=edge_index.device).repeat_interleave(edge_index.shape[1]) * n_nodes
    return edge_index.repeat(1, n_graphs) + shift


class GraphModel(nn.Module):
    """Base class: runs `self.spatial` on every time step and hands (batch, turbines, steps, hidden) on."""

    def __init__(self, edge_index: torch.Tensor, n_nodes: int):
        super().__init__()
        self.register_buffer("edge_index", torch.as_tensor(edge_index, dtype=torch.long))
        self.n_nodes = n_nodes

    def spatial(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        b, n, steps, f = x.shape
        snapshots = x.permute(0, 2, 1, 3).reshape(b * steps * n, f)  # every (sample, step) is one graph
        edges = batch_edges(self.edge_index, n, b * steps)
        h = self.spatial(snapshots, edges)
        return h.reshape(b, steps, n, -1).permute(0, 2, 1, 3)  # (batch, turbines, steps, hidden)


class GCNModel(GraphModel):
    def __init__(self, edge_index, n_nodes: int, in_features: int = 5, hidden: int = 32, steps: int = 12):
        super().__init__(edge_index, n_nodes)
        self.gcn1 = GCNConv(in_features, hidden)
        self.gcn2 = GCNConv(hidden, hidden)
        self.fc = nn.Linear(hidden * steps, 1)

    def spatial(self, h, edge_index):
        return torch.relu(self.gcn2(torch.relu(self.gcn1(h, edge_index)), edge_index))

    def forward(self, x):
        h = self.encode(x)
        return self.fc(h.flatten(2)).squeeze(-1)


class GATModel(GraphModel):
    def __init__(self, edge_index, n_nodes: int, in_features: int = 5, hidden: int = 32, steps: int = 12):
        super().__init__(edge_index, n_nodes)
        self.gat1 = GATConv(in_features, hidden, heads=4)
        self.gat2 = GATConv(hidden * 4, hidden)
        self.fc = nn.Linear(hidden * steps, 1)

    def spatial(self, h, edge_index):
        return torch.relu(self.gat2(torch.relu(self.gat1(h, edge_index)), edge_index))

    def forward(self, x):
        h = self.encode(x)
        return self.fc(h.flatten(2)).squeeze(-1)


class GCNLSTMModel(GraphModel):
    def __init__(self, edge_index, n_nodes: int, in_features: int = 5, hidden: int = 32):
        super().__init__(edge_index, n_nodes)
        self.gcn = GCNConv(in_features, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def spatial(self, h, edge_index):
        return torch.relu(self.gcn(h, edge_index))

    def forward(self, x):
        h = self.encode(x)
        b, n, steps, hidden = h.shape
        out, _ = self.lstm(h.reshape(b * n, steps, hidden))
        return self.fc(out[:, -1]).reshape(b, n)


class GATLSTMModel(GraphModel):
    def __init__(self, edge_index, n_nodes: int, in_features: int = 5, hidden: int = 32):
        super().__init__(edge_index, n_nodes)
        self.gat = GATConv(in_features, hidden, heads=2)
        self.lstm = nn.LSTM(hidden * 2, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def spatial(self, h, edge_index):
        return torch.relu(self.gat(h, edge_index))

    def forward(self, x):
        h = self.encode(x)
        b, n, steps, width = h.shape
        out, _ = self.lstm(h.reshape(b * n, steps, width))
        return self.fc(out[:, -1]).reshape(b, n)


class GATTransformerModel(GraphModel):
    """Graph attention per time step, then self-attention across the time steps."""

    def __init__(self, edge_index, n_nodes: int, in_features: int = 5, hidden: int = 32, steps: int = 12):
        super().__init__(edge_index, n_nodes)
        self.gat = GATConv(in_features, hidden, heads=2)
        self.block = TransformerBlock(n_nodes * hidden * 2, n_nodes, steps)

    def spatial(self, h, edge_index):
        return torch.relu(self.gat(h, edge_index))

    def forward(self, x):
        h = self.encode(x)  # (batch, turbines, steps, width)
        b, n, steps, width = h.shape
        return self.block(h.permute(0, 2, 1, 3).reshape(b, steps, n * width))
