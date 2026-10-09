"""Train one model on the training period and score it on the test period."""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch

from .data import Dataset
from .metrics import compute_metrics
from .models import build_model


@dataclass
class TrainConfig:
    epochs: int = 30
    lr: float = 1e-3
    batch_size: int = 64   # time steps per batch; every batch holds all turbines of those steps
    patience: int = 5      # stop after this many epochs without a better validation loss


def _tensors(ds: Dataset, times: np.ndarray):
    x, y, mask = ds.batch(times)
    return (torch.from_numpy(np.ascontiguousarray(x)),
            torch.from_numpy(np.nan_to_num(y)),
            torch.from_numpy(mask))


def masked_squared_error(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sum of squared errors over the (step, turbine) pairs that can be scored."""
    return (((pred - target) ** 2) * mask).sum()


@torch.no_grad()
def validation_loss(model: torch.nn.Module, ds: Dataset, times: np.ndarray, batch_size: int = 256) -> float:
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(times), batch_size):
        x, y, mask = _tensors(ds, times[i:i + batch_size])
        total += masked_squared_error(model(x), y, mask).item()
        count += int(mask.sum())
    return total / max(count, 1)


@torch.no_grad()
def predict(model: torch.nn.Module, ds: Dataset, times: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Forecasts in kW, shape (len(times), N)."""
    model.eval()
    out = [model(_tensors(ds, times[i:i + batch_size])[0]).numpy() for i in range(0, len(times), batch_size)]
    return ds.to_kw(np.concatenate(out))


def fit(model: torch.nn.Module, ds: Dataset, seed: int, cfg: TrainConfig, log=print) -> dict:
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    best, best_state, best_epoch, waited = float("inf"), None, 0, 0
    started = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = rng.permutation(ds.train)
        train_sum, train_count = 0.0, 0
        for i in range(0, len(order), cfg.batch_size):
            x, y, mask = _tensors(ds, order[i:i + cfg.batch_size])
            optimizer.zero_grad()
            sse = masked_squared_error(model(x), y, mask)
            (sse / mask.sum().clamp(min=1)).backward()
            optimizer.step()
            train_sum += sse.item()
            train_count += int(mask.sum())

        val = validation_loss(model, ds, ds.val)
        log(f"  epoch {epoch:2d}  train MSE {train_sum / train_count:.4f}  val MSE {val:.4f}")
        if val < best - 1e-5:
            best, best_state, best_epoch, waited = val, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            waited += 1
            if waited >= cfg.patience:
                break

    model.load_state_dict(best_state)
    return {"best_epoch": best_epoch, "epochs_run": epoch, "val_mse_standardised": best,
            "train_seconds": round(time.time() - started, 1)}


def score(ds: Dataset, pred_kw: np.ndarray, times: np.ndarray) -> dict[str, float]:
    """Metrics in kW over the eligible (step, turbine) pairs; forecasts below 0 kW are clipped to 0."""
    mask = ds.eligible[times]
    return compute_metrics(ds.grid.y[times][mask], np.clip(pred_kw[mask], 0.0, None))


def run_model(name: str, ds: Dataset, edge_index: np.ndarray, seed: int, cfg: TrainConfig, log=print):
    """Train `name` with `seed`; returns (result dict, test forecasts in kW)."""
    torch.manual_seed(seed)
    model = build_model(name, edge_index, len(ds.grid.turbine_ids))
    n_params = sum(p.numel() for p in model.parameters())
    log(f"{name} (seed {seed}, {n_params:,} parameters)")

    info = fit(model, ds, seed, cfg, log)
    pred = predict(model, ds, ds.test)
    result = {"model": name, "seed": seed, "parameters": n_params, **info, **score(ds, pred, ds.test)}
    return result, pred
