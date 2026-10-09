"""The eight forecasting models behind one factory."""
from __future__ import annotations

import torch.nn as nn

from .graph import GATLSTMModel, GATModel, GATTransformerModel, GCNLSTMModel, GCNModel
from .temporal import LSTMModel, TCNModel, TransformerModel

MODEL_NAMES = ["LSTM", "TCN", "Transformer", "GCN", "GAT", "GCN-LSTM", "GAT-LSTM", "GAT-Transformer"]
GRAPH_MODELS = {"GCN", "GAT", "GCN-LSTM", "GAT-LSTM", "GAT-Transformer"}


def build_model(name: str, edge_index, n_nodes: int, in_features: int = 5, steps: int = 12) -> nn.Module:
    if name == "LSTM":
        return LSTMModel(in_features)
    if name == "TCN":
        return TCNModel(in_features)
    if name == "Transformer":
        return TransformerModel(n_nodes, in_features, steps)
    if name == "GCN":
        return GCNModel(edge_index, n_nodes, in_features, steps=steps)
    if name == "GAT":
        return GATModel(edge_index, n_nodes, in_features, steps=steps)
    if name == "GCN-LSTM":
        return GCNLSTMModel(edge_index, n_nodes, in_features)
    if name == "GAT-LSTM":
        return GATLSTMModel(edge_index, n_nodes, in_features)
    if name == "GAT-Transformer":
        return GATTransformerModel(edge_index, n_nodes, in_features, steps=steps)
    raise ValueError(f"Unknown model '{name}'. Choose from {MODEL_NAMES}.")
