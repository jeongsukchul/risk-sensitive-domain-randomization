from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_METRICS = [
    "target4/p",
    "target4/p_base",
    "target4/p_ema",
    "target4/lambda",
    "target4/sample_mean",
    "target4/forward_kl",
    "target4/reverse_kl",
    "target4/wasserstein",
    "target4/sinkhorn",
    "target4/ess",
    "target4/energy_w2",
    "target4/target_mean",
    "target4/optimal_p",
]


def _load_history(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {key: np.asarray(data[key], dtype=np.float64).reshape(-1) for key in data.files}


def _plot_metric(ax, metric_name: str, gbs_values: np.ndarray, gmmvi_values: np.ndarray) -> None:
    gbs_steps = np.arange(gbs_values.size)
    gmmvi_steps = np.arange(gmmvi_values.size)
    ax.plot(gbs_steps, gbs_values, label="GBS", color="tab:blue", linewidth=1.8)
    ax.plot(gmmvi_steps, gmmvi_values, label="GMMVI", color="tab:orange", linewidth=1.8)
    ax.set_title(metric_name.replace("target4/", ""))
    ax.set_xlabel("iteration")
    ax.grid(alpha=0.3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay GBS and GMMVI toy history metrics.")
    parser.add_argument(
        "--gbs-history",
        type=Path,
        default=Path("results/target4_history.npz"),
    )
    parser.add_argument(
        "--gmmvi-history",
        type=Path,
        default=Path("results/gmmvi_target4_history.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/gbs_vs_gmmvi_history_metrics.png"),
    )
    parser.add_argument("--cols", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gbs_hist = _load_history(args.gbs_history)
    gmmvi_hist = _load_history(args.gmmvi_history)

    metrics = [
        metric
        for metric in DEFAULT_METRICS
        if metric in gbs_hist and metric in gmmvi_hist
    ]
    if not metrics:
        raise ValueError("No shared metrics found between the two history files.")

    cols = max(1, args.cols)
    rows = math.ceil(len(metrics) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.5 * rows), squeeze=False)
    axes_flat = axes.reshape(-1)

    for idx, metric in enumerate(metrics):
        ax = axes_flat[idx]
        _plot_metric(ax, metric, gbs_hist[metric], gmmvi_hist[metric])

    for idx in range(len(metrics), len(axes_flat)):
        axes_flat[idx].axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("GBS vs GMMVI Training Metrics", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.as_posix(), dpi=160)
    plt.close(fig)
    print(f"Saved plot to: {args.output}")


if __name__ == "__main__":
    main()
