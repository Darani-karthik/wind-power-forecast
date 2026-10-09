"""The turbine graph: each turbine listens to its k nearest neighbours."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph


def node_coordinates(locations: pd.DataFrame, turbine_ids: list[int]) -> np.ndarray:
    """(N, 2) x / y of the given turbines in that order, so node i of the graph is node i of the data."""
    indexed = locations.set_index("TurbID")
    missing = [tid for tid in turbine_ids if tid not in indexed.index]
    if missing:
        raise ValueError(f"No coordinates for turbines {missing}.")
    return indexed.loc[turbine_ids, ["x", "y"]].to_numpy(dtype=float)


def knn_edges(coords: np.ndarray, k: int = 5) -> np.ndarray:
    """edge_index (2, N * k): row 0 is the sending neighbour, row 1 the receiving turbine."""
    adjacency = kneighbors_graph(coords, n_neighbors=k, mode="connectivity", include_self=False).tocoo()
    return np.stack([adjacency.col, adjacency.row]).astype(np.int64)  # A[i, j] = 1 means j is a neighbour of i


def turbine_graph(locations: pd.DataFrame, turbine_ids: list[int], k: int = 5) -> np.ndarray:
    return knn_edges(node_coordinates(locations, turbine_ids), k)
