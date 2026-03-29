from __future__ import annotations

import argparse
import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from learning.module.gbs.sinkhorn_metrics import (
    energy_wasserstein_1d,
    effective_sample_size_from_log_weights,
    interatomic_wasserstein_1d,
    sinkhorn_distance,
)
from learning.module.gbs.target4_notebook_utils import (
    compute_target4_metrics,
    optimal_p_from_target_mean,
    sample_truncated_exponential,
    target4_logprob,
)


def _load_final_lambda(history_path: Path | None) -> float | None:
    if history_path is None or not history_path.exists():
        return None
    history = np.load(history_path)
    if "target4/lambda" not in history:
        return None
    values = np.asarray(history["target4/lambda"], dtype=np.float64).reshape(-1)
    if values.size == 0:
        return None
    return float(values[-1])


def _safe_n_particles(dim: int, n_particles: int | None, n_spatial_dim: int) -> int | None:
    if n_particles is not None:
        return n_particles
    if dim % n_spatial_dim != 0:
        return None
    return dim // n_spatial_dim


def _target_metrics(
    samples: np.ndarray,
    lam: float,
    num_bins: int,
    sinkhorn_num_samples: int,
    n_particles: int | None,
    n_spatial_dim: int,
    key: jax.Array,
) -> dict[str, float]:
    flat_samples = samples.reshape(-1)
    sample_mean = float(np.mean(flat_samples))
    sample_std = float(np.std(flat_samples))
    sample_min = float(np.min(flat_samples))
    sample_max = float(np.max(flat_samples))
    forward_kl, reverse_kl, wasserstein = compute_target4_metrics(
        flat_samples,
        lam,
        num_bins=num_bins,
        key=key,
    )

    n_sink = min(sinkhorn_num_samples, samples.shape[0])
    key, subkey = jax.random.split(key)
    ref = sample_truncated_exponential(subkey, lam, samples[:n_sink].shape)
    sinkhorn = sinkhorn_distance(jnp.asarray(samples[:n_sink]), ref)
    ess = effective_sample_size_from_log_weights(target4_logprob(jnp.asarray(samples), lam))

    metrics = {
        "lambda": float(lam),
        "sample_mean": sample_mean,
        "sample_std": sample_std,
        "sample_min": sample_min,
        "sample_max": sample_max,
        "forward_kl": float(forward_kl),
        "reverse_kl": float(reverse_kl),
        "wasserstein_to_target": float(wasserstein),
        "sinkhorn_to_target": float(sinkhorn),
        "ess": float(ess),
    }

    energy_w2 = energy_wasserstein_1d(
        jnp.asarray(samples[:n_sink]),
        ref,
        lam,
    )
    metrics["energy_w2"] = float(energy_w2)
    if n_particles is not None and n_particles > 1:
        interatomic_w2 = interatomic_wasserstein_1d(
            jnp.asarray(samples[:n_sink]),
            ref,
            n_particles=n_particles,
            n_spatial_dim=n_spatial_dim,
        )
        metrics["interatomic_w2"] = float(interatomic_w2)

    optimal_p, target_mean = optimal_p_from_target_mean(lam, tau=0.10)
    metrics["target_mean"] = float(target_mean)
    metrics["optimal_p_tau_0.10"] = float(optimal_p)
    return metrics


def _pairwise_metrics(
    gbs_samples: np.ndarray,
    gmmvi_samples: np.ndarray,
    sinkhorn_num_samples: int,
) -> dict[str, float]:
    gbs_flat = np.sort(np.clip(gbs_samples.reshape(-1), 0.0, 1.0))
    gmmvi_flat = np.sort(np.clip(gmmvi_samples.reshape(-1), 0.0, 1.0))
    n_flat = min(gbs_flat.size, gmmvi_flat.size)

    n_rows = min(sinkhorn_num_samples, gbs_samples.shape[0], gmmvi_samples.shape[0])
    return {
        "mean_gap": float(np.mean(gmmvi_samples) - np.mean(gbs_samples)),
        "std_gap": float(np.std(gmmvi_samples) - np.std(gbs_samples)),
        "l1_sorted_flat": float(np.mean(np.abs(gbs_flat[:n_flat] - gmmvi_flat[:n_flat]))),
        "rmse_sorted_flat": float(np.sqrt(np.mean((gbs_flat[:n_flat] - gmmvi_flat[:n_flat]) ** 2))),
        "sinkhorn_gbs_vs_gmmvi": float(
            sinkhorn_distance(
                jnp.asarray(gbs_samples[:n_rows]),
                jnp.asarray(gmmvi_samples[:n_rows]),
            )
        ),
    }


def _write_csv(output_path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare final toy metrics for GBS and GMMVI.")
    parser.add_argument(
        "--gbs-samples",
        type=Path,
        default=Path("learning/module/gbs/results/gbs_samples.npy"),
    )
    parser.add_argument(
        "--gmmvi-samples",
        type=Path,
        default=Path("learning/module/gbs/results/gmmvi_samples.npy"),
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
    parser.add_argument("--gbs-lambda", type=float, default=None)
    parser.add_argument("--gmmvi-lambda", type=float, default=None)
    parser.add_argument("--metric-num-bins", type=int, default=128)
    parser.add_argument("--sinkhorn-num-samples", type=int, default=512)
    parser.add_argument("--n-particles", type=int, default=None)
    parser.add_argument("--n-spatial-dim", type=int, default=1)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("learning/module/gbs/results/gbs_gmmvi_comparison.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gbs_samples = np.load(args.gbs_samples)
    gmmvi_samples = np.load(args.gmmvi_samples)

    if gbs_samples.ndim != 2 or gmmvi_samples.ndim != 2:
        raise ValueError("Expected both sample files to have shape [num_samples, dim].")
    if gbs_samples.shape[1] != gmmvi_samples.shape[1]:
        raise ValueError(
            f"Feature dimensions must match, got {gbs_samples.shape[1]} and {gmmvi_samples.shape[1]}"
        )

    dim = int(gbs_samples.shape[1])
    n_particles = _safe_n_particles(dim, args.n_particles, args.n_spatial_dim)
    gbs_lambda = args.gbs_lambda
    if gbs_lambda is None:
        gbs_lambda = _load_final_lambda(args.gbs_history)
    gmmvi_lambda = args.gmmvi_lambda
    if gmmvi_lambda is None:
        gmmvi_lambda = _load_final_lambda(args.gmmvi_history)
    if gbs_lambda is None or gmmvi_lambda is None:
        raise ValueError("Could not infer lambda from history files. Pass --gbs-lambda and --gmmvi-lambda.")

    key = jax.random.PRNGKey(0)
    key, gbs_key, gmmvi_key = jax.random.split(key, 3)
    gbs_metrics = _target_metrics(
        gbs_samples,
        gbs_lambda,
        args.metric_num_bins,
        args.sinkhorn_num_samples,
        n_particles,
        args.n_spatial_dim,
        gbs_key,
    )
    gmmvi_metrics = _target_metrics(
        gmmvi_samples,
        gmmvi_lambda,
        args.metric_num_bins,
        args.sinkhorn_num_samples,
        n_particles,
        args.n_spatial_dim,
        gmmvi_key,
    )
    pairwise = _pairwise_metrics(gbs_samples, gmmvi_samples, args.sinkhorn_num_samples)

    rows = []
    for metric_name, value in gbs_metrics.items():
        rows.append({"group": "gbs", "metric": metric_name, "value": value})
    for metric_name, value in gmmvi_metrics.items():
        rows.append({"group": "gmmvi", "metric": metric_name, "value": value})
    for metric_name, value in pairwise.items():
        rows.append({"group": "pairwise", "metric": metric_name, "value": value})
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_csv, rows)

    print("GBS metrics")
    for metric_name, value in gbs_metrics.items():
        print(f"  {metric_name}: {value:.6f}")
    print("GMMVI metrics")
    for metric_name, value in gmmvi_metrics.items():
        print(f"  {metric_name}: {value:.6f}")
    print("Pairwise comparison")
    for metric_name, value in pairwise.items():
        print(f"  {metric_name}: {value:.6f}")
    print(f"Saved CSV to: {args.output_csv}")


if __name__ == "__main__":
    main()
