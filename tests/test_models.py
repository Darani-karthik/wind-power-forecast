import numpy as np
import pytest
import torch

from src.data import SEQ_LEN
from src.graph import knn_edges
from src.models import MODEL_NAMES, build_model
from src.models.graph import batch_edges
from src.models.temporal import CausalConv, TCNModel

PAIR = np.array([[1, 0], [0, 1]])  # turbines 0 and 1 listen to each other; turbine 2 is isolated


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_gives_one_forecast_per_turbine(name, batch):
    edges = knn_edges(np.random.default_rng(0).random((6, 2)), k=2)
    model = build_model(name, edges, n_nodes=6)
    out = model(torch.randn(batch, 6, SEQ_LEN, 5))
    assert out.shape == (batch, 6)
    assert torch.isfinite(out).all()

    out.sum().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_unknown_model_is_an_error():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("Prophet", PAIR, n_nodes=3)


def test_tcn_uses_every_step_of_the_window():
    # Regression test: the prototype's TCN padded both sides and read the last position, so only the
    # final input step reached the output (the gradient to the other 11 steps was exactly zero).
    torch.manual_seed(0)
    x = torch.randn(8, 3, SEQ_LEN, 5, requires_grad=True)
    TCNModel()(x).sum().backward()
    per_step = x.grad.abs().sum(dim=(0, 1, 3))
    assert (per_step > 0).all()


def test_causal_conv_ignores_later_inputs():
    conv = CausalConv(3, 4, kernel=3, dilation=2).eval()
    a = torch.randn(1, 3, 12)
    b = a.clone()
    b[:, :, 8:] = torch.randn(1, 3, 4)
    assert torch.allclose(conv(a)[:, :, :8], conv(b)[:, :, :8])


@pytest.mark.parametrize("name", ["GCN", "GAT", "GCN-LSTM", "GAT-LSTM"])
def test_graph_models_only_listen_to_neighbours(name):
    torch.manual_seed(0)
    model = build_model(name, PAIR, n_nodes=3).eval()
    x = torch.randn(2, 3, SEQ_LEN, 5)
    base = model(x)

    far = x.clone()
    far[:, 2] += 5.0  # turbine 2 is not connected to turbine 0
    near = x.clone()
    near[:, 1] += 5.0  # turbine 1 is
    assert torch.allclose(model(far)[:, 0], base[:, 0], atol=1e-6)
    assert not torch.allclose(model(near)[:, 0], base[:, 0], atol=1e-4)


def test_batch_edges_copies_the_graph_with_shifted_ids():
    out = batch_edges(torch.tensor(PAIR), n_nodes=3, n_graphs=2)
    assert out.tolist() == [[1, 0, 4, 3], [0, 1, 3, 4]]
