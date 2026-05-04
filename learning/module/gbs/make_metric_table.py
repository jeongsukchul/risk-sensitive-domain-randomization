from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TargetSpec:
    key: str
    label: str
    beta: float


TARGET_SPECS = (
    TargetSpec("target_A", r"Tilted Linear (\(\beta=-5\))", -5.0),
    TargetSpec("target_B", r"Fixed-Mode Cosine (\(\beta=-10\))", -10.0),
    TargetSpec("target_C", r"Policy-Shifted Cosine (\(\beta=-10\))", -10.0),
)

ALGORITHM_SPECS = (
    ("gmmvi", "GMMVI"),
    ("dis_lv", "DIS-LV"),
)

METRIC_COLUMNS = (
    ("target__p", r"p_u"),
    ("target__sinkhorn", "Sinkhorn"),
    ("target__energy_w2", "Energy W2"),
)

INITIAL_POLICY_UPDATE_WINDOW = 5.0
SEED_DIR_PATTERN = re.compile(r"^seed_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read raw metric CSV logs from algorithm_comparison.py and emit a LaTeX "
            "table with seed-aggregated initial-window-average-to-final metric values."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("compare_GBS_GMMVI"),
        help="Directory containing target_*_dim*_beta* result folders, optionally with seed_*/raw_metrics subfolders.",
    )
    parser.add_argument(
        "--dims",
        type=int,
        nargs="+",
        default=[2, 5, 10, 20],
        help="Latent dimensions to include for each target.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the LaTeX table. The table is always printed.",
    )
    parser.add_argument(
        "--caption",
        default=(
            "Toy-MDP tracking results across target families and latent dimensions. "
            "Each entry reports mean $\\pm$ standard deviation across seeds for the "
            "average over the first 5 policy updates $\\to$ final values from the metric logs. "
            "Lower Sinkhorn distance and lower Energy W2 indicate better agreement "
            "with the analytically sampled target distribution."
        ),
    )
    parser.add_argument("--label", default="tab:toy-results")
    return parser.parse_args()


def _float_token(value: float) -> str:
    text = f"{value:.1f}"
    return text[:-2] if text.endswith(".0") else text


def _target_dir(results_dir: Path, target: TargetSpec, dim: int) -> Path:
    exact = results_dir / f"{target.key}_dim{dim}_beta{target.beta}"
    if exact.exists():
        return exact

    beta_pattern = re.escape(_float_token(target.beta))
    pattern = re.compile(rf"^{re.escape(target.key)}_dim{dim}_beta{beta_pattern}(?:\.0+)?$")
    matches = sorted(path for path in results_dir.iterdir() if path.is_dir() and pattern.match(path.name))
    if not matches:
        raise FileNotFoundError(
            f"Missing results directory for {target.key}, dim={dim}, beta={target.beta:g} under {results_dir}"
        )
    return matches[-1]


def _read_metric_csv(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing metric file: {path}")
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    return np.atleast_1d(data)


def _seed_sort_key(path: Path) -> tuple[int, str]:
    match = SEED_DIR_PATTERN.fullmatch(path.name)
    if match is None:
        return (10**9, path.name)
    return (int(match.group(1)), path.name)


def _metric_paths_for_algorithm(target_dir: Path, algo_key: str) -> list[Path]:
    metric_relpath = Path("raw_metrics") / f"{algo_key}_metrics.csv"
    seed_dirs = sorted(
        (
            path
            for path in target_dir.iterdir()
            if path.is_dir() and SEED_DIR_PATTERN.fullmatch(path.name)
        ),
        key=_seed_sort_key,
    )
    if seed_dirs:
        metric_paths = [seed_dir / metric_relpath for seed_dir in seed_dirs]
        missing_paths = [path for path in metric_paths if not path.exists()]
        if missing_paths:
            missing = ", ".join(path.parent.parent.name for path in missing_paths)
            raise FileNotFoundError(
                f"Missing seed metric files for {algo_key} under {target_dir}: {missing}"
            )
        return metric_paths

    legacy_path = target_dir / metric_relpath
    if legacy_path.exists():
        return [legacy_path]
    raise FileNotFoundError(f"Missing metric file: {legacy_path}")


def _initial_average_value(data: np.ndarray, column: str) -> float:
    if column not in data.dtype.names:
        raise KeyError(f"Column {column!r} not found in metric file")
    values = np.asarray(data[column], dtype=np.float64).reshape(-1)
    if "policy_update_steps" in data.dtype.names:
        update_steps = np.asarray(data["policy_update_steps"], dtype=np.float64).reshape(-1)
        window_mask = (update_steps > 0.0) & (update_steps <= INITIAL_POLICY_UPDATE_WINDOW)
        window_values = values[window_mask]
    else:
        window_values = values[1:11]
    finite = window_values[np.isfinite(window_values)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _last_finite_value(data: np.ndarray, column: str) -> float:
    if column not in data.dtype.names:
        raise KeyError(f"Column {column!r} not found in metric file")
    values = np.asarray(data[column], dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(finite[-1])


def _format_number(value: float) -> str:
    if not math.isfinite(value):
        return r"\mathrm{nan}"
    abs_value = abs(value)
    if abs_value == 0.0:
        return "0"
    if abs_value < 1e-3 or abs_value >= 1e3:
        exponent = int(math.floor(math.log10(abs_value)))
        mantissa = value / (10.0**exponent)
        return rf"{mantissa:.2g} \times 10^{{{exponent}}}"
    if abs_value < 0.01:
        return f"{value:.3g}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _metric_cells(results_dir: Path, target: TargetSpec, dim: int, algo_key: str) -> list[str]:
    metric_paths = _metric_paths_for_algorithm(_target_dir(results_dir, target, dim), algo_key)
    seed_data = [_read_metric_csv(metric_path) for metric_path in metric_paths]
    cells = []
    for column, _ in METRIC_COLUMNS:
        initial_values = np.asarray(
            [_initial_average_value(data, column) for data in seed_data],
            dtype=np.float64,
        )
        final_values = np.asarray(
            [_last_finite_value(data, column) for data in seed_data],
            dtype=np.float64,
        )
        initial_finite = initial_values[np.isfinite(initial_values)]
        final_finite = final_values[np.isfinite(final_values)]
        initial_mean = float(np.mean(initial_finite)) if initial_finite.size else float("nan")
        initial_std = float(np.std(initial_finite)) if initial_finite.size else float("nan")
        final_mean = float(np.mean(final_finite)) if final_finite.size else float("nan")
        final_std = float(np.std(final_finite)) if final_finite.size else float("nan")
        cells.append(
            rf"\(({_format_number(initial_mean)} \pm {_format_number(initial_std)}) "
            rf"\to ({_format_number(final_mean)} \pm {_format_number(final_std)})\)"
        )
    return cells


def build_table(results_dir: Path, dims: list[int], caption: str, label: str) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{@{}lllccc@{}}",
        r"\toprule",
        r"Target & \(d\) & Method & \(p_u\) & Sinkhorn & Energy W2 \\",
        r"\midrule",
    ]

    for target_index, target in enumerate(TARGET_SPECS):
        target_rows = len(dims) * len(ALGORITHM_SPECS)
        for dim_index, dim in enumerate(dims):
            for algo_index, (algo_key, algo_label) in enumerate(ALGORITHM_SPECS):
                target_cell = rf"\multirow{{{target_rows}}}{{*}}{{{target.label}}}" if dim_index == 0 and algo_index == 0 else ""
                dim_cell = str(dim) if algo_index == 0 else ""
                row_cells = [target_cell, dim_cell, algo_label]
                row_cells.extend(_metric_cells(results_dir, target, dim, algo_key))
                lines.append(" & ".join(row_cells) + r" \\")
        if target_index != len(TARGET_SPECS) - 1:
            lines.append(r"\addlinespace[0.25em]")
            lines.append("")
            lines.append(r"\midrule")
            lines.append("")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    table = build_table(
        results_dir=args.results_dir,
        dims=args.dims,
        caption=args.caption,
        label=args.label,
    )
    print(table)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(table + "\n")


if __name__ == "__main__":
    main()
