"""Figures: model comparison, an example forecast and the turbine graph (light surface, fixed palette)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, DIM = "#e1e0d9", "#c3c2b7", "#b9b8b0"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1 and 2

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.linewidth": 1.0, "axes.grid": False,
})


TOP_MARGIN_IN = 0.95  # room above the axes for the two title lines


def _title(fig, title: str, subtitle: str):
    height = fig.get_figheight()
    fig.text(0.02, 1 - 0.15 / height, title, fontsize=13, fontweight="semibold", color=INK, ha="left", va="top")
    fig.text(0.02, 1 - 0.47 / height, subtitle, fontsize=9, color=INK2, ha="left", va="top")


def _margins(fig, left: float, right: float, bottom_in: float):
    height = fig.get_figheight()
    fig.subplots_adjust(left=left, right=right, top=1 - TOP_MARGIN_IN / height, bottom=bottom_in / height)


def _clean(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0, labelsize=9)


def plot_model_comparison(rows: list[dict], subtitle: str, path) -> None:
    """Horizontal bars of test RMSE; neural models in blue, reference forecasts in gray."""
    rows = sorted(rows, key=lambda r: r["RMSE"])
    n = len(rows)
    fig, ax = plt.subplots(figsize=(7.4, 0.42 * n + 1.9))
    _margins(fig, left=0.27, right=0.95, bottom_in=0.7)

    y = np.arange(n)[::-1]
    for yi, row in zip(y, rows):
        is_model = row["kind"] == "model"
        ax.barh(yi, row["RMSE"], height=0.56, color=BLUE if is_model else DIM, zorder=3)
        end = row["RMSE"]
        if row.get("RMSE_std"):
            ax.plot([end - row["RMSE_std"], end + row["RMSE_std"]], [yi, yi], color=INK2, lw=1, zorder=4)
            end += row["RMSE_std"]
        ax.text(end + 4, yi, f"{row['RMSE']:.1f}", va="center", ha="left", fontsize=9, color=INK2)

    ax.set_yticks(y, [r["model"] for r in rows], fontsize=9.5, color=INK)
    ax.set_xlim(0, max(r["RMSE"] + (r.get("RMSE_std") or 0) for r in rows) * 1.14)
    ax.xaxis.grid(True, color=GRID, lw=1, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("Test RMSE (kW)", fontsize=9)
    _clean(ax)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=DIM)]
    ax.legend(handles, ["Neural model (mean ± std over seeds)", "Reference forecast"], loc="upper right",
              frameon=False, fontsize=8.5, labelcolor=INK2, handlelength=1.0, handleheight=1.0)
    _title(fig, "Forecast error on the held-out test period (lower is better)", subtitle)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_forecast_example(ds, forecast_kw: np.ndarray, model: str, path, days: float = 4.0) -> None:
    """Measured vs forecast power of one turbine over the first days of the test period."""
    times = ds.test
    n_steps = int(days * 24 * 3600 / ds.grid.step.total_seconds())
    window = np.arange(len(times))[:n_steps]
    mask = ds.eligible[times][window]
    # a representative turbine: the median output among those with few gaps (not the best- or worst-looking one)
    counts = mask.sum(axis=0)
    output = np.array([ds.grid.y[times[window]][mask[:, n], n].mean() if mask[:, n].any() else -1.0
                       for n in range(mask.shape[1])])
    few_gaps = np.flatnonzero(counts >= 0.9 * counts.max())
    turbine = int(few_gaps[np.argsort(output[few_gaps])[len(few_gaps) // 2]])

    actual = np.where(mask[:, turbine], ds.grid.y[times[window], turbine], np.nan)
    pred = np.where(mask[:, turbine], np.clip(forecast_kw[window, turbine], 0, None), np.nan)
    stamps = ds.grid.times[times[window]]

    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    _margins(fig, left=0.1, right=0.97, bottom_in=0.5)
    ax.plot(stamps, actual, color=BLUE, lw=1.6, solid_capstyle="round", label="Measured", zorder=3)
    ax.plot(stamps, pred, color=ORANGE, lw=1.6, solid_capstyle="round", label=f"{model} forecast", zorder=4)
    ax.yaxis.grid(True, color=GRID, lw=1, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylabel("Active power (kW)", fontsize=9)
    ax.set_ylim(0, None)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    _clean(ax)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK2, ncol=2, handlelength=1.6)
    turbine_id = ds.grid.turbine_ids[turbine]
    _title(fig, f"{model}: 15-minute-ahead forecast vs measurement",
           f"Turbine {turbine_id} (median output of the turbines), first {days:g} days of the test period")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_turbine_graph(locations, turbine_ids: list[int], edge_index: np.ndarray, k: int, path) -> None:
    """The farm layout with the neighbour graph; one turbine and its k senders highlighted."""
    indexed = locations.set_index("TurbID")
    coords = indexed.loc[turbine_ids, ["x", "y"]].to_numpy(float)
    others = indexed.drop(index=turbine_ids)[["x", "y"]].to_numpy(float)

    centre = coords.mean(axis=0)
    focal = int(np.linalg.norm(coords - centre, axis=1).argmin())
    senders = edge_index[0][edge_index[1] == focal]

    fig, ax = plt.subplots(figsize=(5.6, 7.6))
    _margins(fig, left=0.13, right=0.97, bottom_in=1.2)
    ax.scatter(others[:, 0], others[:, 1], s=14, color="#d3d2ca", zorder=1, label="Turbines not used")
    for src, dst in edge_index.T:
        ax.plot(*coords[[src, dst]].T, color=AXIS, lw=0.8, zorder=2)
    for src in senders:
        ax.plot(*coords[[src, focal]].T, color=BLUE, lw=1.8, zorder=3)
    ax.scatter(coords[:, 0], coords[:, 1], s=34, color=DIM, edgecolor=SURFACE, linewidth=1.5, zorder=4,
               label="Modelled turbines")
    ax.scatter(coords[senders, 0], coords[senders, 1], s=48, color=BLUE, edgecolor=SURFACE, linewidth=1.5,
               zorder=5, label=f"Its {k} nearest neighbours")
    ax.scatter(*coords[focal], s=78, color=ORANGE, edgecolor=SURFACE, linewidth=1.5, zorder=6,
               label=f"Example turbine ({turbine_ids[focal]})")

    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=9)
    ax.set_ylabel("y (m)", fontsize=9)
    _clean(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, frameon=False, fontsize=8.5,
              labelcolor=INK2, markerscale=0.9)
    _title(fig, "Each turbine listens to its nearest neighbours",
           f"{len(turbine_ids)} modelled turbines of the wind farm, edges point to the receiving turbine")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
