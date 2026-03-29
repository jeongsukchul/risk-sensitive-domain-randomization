from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_history(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {key: np.asarray(data[key], dtype=np.float64).reshape(-1) for key in data.files}


def _metric_values(history: dict[str, np.ndarray], metric: str, hide_first: bool) -> np.ndarray:
    if metric not in history:
        raise KeyError(f"Missing metric '{metric}' in history file.")
    values = history[metric].copy()
    if hide_first and values.size:
        values[0] = np.nan
    return values


def _resolve_history(path: Path, pattern: str) -> Path:
    if path.exists():
        return path
    matches = sorted(path.parent.glob(pattern))
    if matches:
        return matches[-1]
    raise FileNotFoundError(f"Could not find history file: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a unified Sinkhorn / Energy W2 comparison plot for GBS and GMMVI target4 runs."
    )
    parser.add_argument(
        "--gbs-history",
        type=Path,
        default=Path("learning/module/gbs/results/target4_history.npz"),
    )
    parser.add_argument(
        "--gmmvi-history",
        type=Path,
        default=Path("learning/module/gbs/results/gmmvi_target4_history.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("learning/module/gbs/results/target4_sinkhorn_energy_comparison.png"),
    )
    parser.add_argument(
        "--hide-initial-point",
        action="store_true",
        help="Hide the first point for plotted metrics, useful when iteration 0 is a placeholder.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gbs_history_path = _resolve_history(args.gbs_history, "target4_history*.npz")
    gmmvi_history_path = _resolve_history(args.gmmvi_history, "gmmvi_target4_history*.npz")

    gbs_history = _load_history(gbs_history_path)
    gmmvi_history = _load_history(gmmvi_history_path)

    metrics = [
        ("target4/sinkhorn", "Sinkhorn Distance"),
        ("target4/energy_w2", "Energy W2"),
    ]
    colors = {"GBS": "tab:blue", "GMMVI": "tab:orange"}
    histories = {"GBS": gbs_history, "GMMVI": gmmvi_history}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=False)

    for ax, (metric_key, title) in zip(axes, metrics):
        for label, history in histories.items():
            values = _metric_values(history, metric_key, hide_first=args.hide_initial_point)
            steps = np.arange(values.size)
            ax.plot(steps, values, label=label, color=colors[label], linewidth=2.0)
        ax.set_title(title)
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("metric value")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(
        "Unified Comparison: GBS vs GMMVI on Target4",
        fontsize=13,
        y=0.98,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.as_posix(), dpi=160)
    plt.close(fig)

    print(f"GBS history: {gbs_history_path}")
    print(f"GMMVI history: {gmmvi_history_path}")
    print(f"Saved comparison plot to: {args.output}")


if __name__ == "__main__":
    main()
