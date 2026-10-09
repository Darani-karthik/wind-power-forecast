import numpy as np
import pandas as pd
import pytest

from src.graph import knn_edges, node_coordinates, turbine_graph


def test_every_turbine_receives_k_edges():
    coords = np.random.default_rng(0).random((12, 2))
    edges = knn_edges(coords, k=3)
    assert edges.shape == (2, 12 * 3)
    assert (np.bincount(edges[1], minlength=12) == 3).all()
    assert (edges[0] != edges[1]).all()  # no self edges


def test_row_zero_sends_row_one_receives():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]])  # A, B, C: C's nearest neighbour is B
    edges = {tuple(e) for e in knn_edges(coords, k=1).T}
    assert (1, 2) in edges      # B sends to C
    assert (2, 1) not in edges  # C is not B's nearest neighbour


def test_nodes_follow_the_requested_turbine_order():
    # Regression test: the prototype built the graph from TurbID 1-30 but fed it data of other turbines.
    locations = pd.DataFrame({"TurbID": [1, 2, 3, 4], "x": [0.0, 100.0, 1.0, 101.0], "y": 0.0})
    coords = node_coordinates(locations, [2, 4, 1, 3])
    np.testing.assert_array_equal(coords[:, 0], [100.0, 101.0, 0.0, 1.0])

    edges = {tuple(e) for e in turbine_graph(locations, [2, 4, 1, 3], k=1).T}
    assert edges == {(1, 0), (0, 1), (3, 2), (2, 3)}  # node 0 (id 2) pairs with node 1 (id 4), id 1 with id 3


def test_unknown_turbine_is_an_error():
    locations = pd.DataFrame({"TurbID": [1, 2], "x": [0.0, 1.0], "y": 0.0})
    with pytest.raises(ValueError, match="99"):
        node_coordinates(locations, [1, 99])
