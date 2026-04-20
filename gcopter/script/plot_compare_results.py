#!/usr/bin/env python3

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot GCOPTER compare metrics and trajectory shapes."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory containing trajectory_metrics.csv and request trajectory csv files.",
    )
    parser.add_argument(
        "--request-id",
        type=int,
        required=True,
        help="Request id used to load request_<id>_trajectory.csv.",
    )
    return parser.parse_args()


def setup_matplotlib(output_dir):
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for style_name in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
        try:
            plt.style.use(style_name)
            break
        except OSError:
            continue
    return plt


def to_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def load_csv_rows(csv_path):
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def solver_order(solvers):
    preferred = ["LBFGS", "IGO", "META", "IGO_XSPACE", "IGO_HEURISTIC"]
    ordered = [solver for solver in preferred if solver in solvers]
    extras = sorted(solver for solver in solvers if solver not in preferred)
    return ordered + extras


def clean_mean(values):
    cleaned = [value for value in values if not math.isnan(value)]
    if not cleaned:
        return math.nan
    return float(np.mean(np.asarray(cleaned, dtype=float)))


def compute_solver_means(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["solver"]].append(row)

    metrics = [
        "objective",
        "wall_time_sec",
        "optimization_time_sec",
        "frontend_time_sec",
        "path_search_time_sec",
        "corridor_generation_time_sec",
        "total_duration_sec",
        "trajectory_length",
        "collision_length",
        "max_collision_distance",
        "max_violation",
        "max_acceleration_violation",
        "max_speed",
        "max_acceleration",
        "acceleration_energy",
        "jerk_energy",
        "min_obstacle_distance",
    ]

    summary_rows = []
    for solver in solver_order(grouped.keys()):
        solver_rows = grouped[solver]
        row = {
            "solver": solver,
            "sample_count": len(solver_rows),
            "success_rate": clean_mean([to_float(item.get("success", "0")) for item in solver_rows]),
        }
        for metric in metrics:
            row[metric] = clean_mean([to_float(item.get(metric, "")) for item in solver_rows])
        summary_rows.append(row)

    lbfgs_time = math.nan
    lbfgs_opt_time = math.nan
    for row in summary_rows:
        if row["solver"] == "LBFGS":
            lbfgs_time = row.get("wall_time_sec", math.nan)
            lbfgs_opt_time = row.get("optimization_time_sec", math.nan)
            break
    for row in summary_rows:
        wall_time = row.get("wall_time_sec", math.nan)
        opt_time = row.get("optimization_time_sec", math.nan)
        if not math.isnan(wall_time) and not math.isnan(lbfgs_time) and abs(lbfgs_time) > 1.0e-12:
            row["wall_time_ratio_vs_lbfgs"] = wall_time / lbfgs_time
        else:
            row["wall_time_ratio_vs_lbfgs"] = math.nan
        if not math.isnan(opt_time) and not math.isnan(lbfgs_opt_time) and abs(lbfgs_opt_time) > 1.0e-12:
            row["optimization_time_ratio_vs_lbfgs"] = opt_time / lbfgs_opt_time
        else:
            row["optimization_time_ratio_vs_lbfgs"] = math.nan

    return summary_rows


def write_solver_mean_csv(summary_rows, output_dir):
    path = output_dir / "solver_mean_metrics.csv"
    fieldnames = [
        "solver",
        "sample_count",
        "success_rate",
        "objective",
        "wall_time_sec",
        "optimization_time_sec",
        "frontend_time_sec",
        "path_search_time_sec",
        "corridor_generation_time_sec",
        "wall_time_ratio_vs_lbfgs",
        "optimization_time_ratio_vs_lbfgs",
        "total_duration_sec",
        "trajectory_length",
        "collision_length",
        "max_collision_distance",
        "max_violation",
        "max_acceleration_violation",
        "max_speed",
        "max_acceleration",
        "acceleration_energy",
        "jerk_energy",
        "min_obstacle_distance",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def plot_mean_metrics(plt, summary_rows, output_dir):
    metrics = [
        ("Mean Benchmark Time", "wall_time_sec", "seconds"),
        ("Mean Optimization Time", "optimization_time_sec", "seconds"),
        ("Mean Frontend Time", "frontend_time_sec", "seconds"),
        ("Mean Flight Duration", "total_duration_sec", "seconds"),
        ("Mean Max Violation", "max_violation", "value"),
        ("Mean Jerk Energy", "jerk_energy", "integral"),
    ]

    solvers = [row["solver"] for row in summary_rows]
    if not solvers:
        return

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle("GCOPTER Compare Mean Metrics", fontsize=18)

    for ax, (title, key, ylabel) in zip(axes.flatten(), metrics):
        values = [to_float(row.get(key, "")) for row in summary_rows]
        x = np.arange(len(solvers))
        palette = {
            "LBFGS": "#0073F0",
            "IGO": "#FF7A00",
            "META": "#4C9F70",
            "IGO_XSPACE": "#9467BD",
            "IGO_HEURISTIC": "#8C564B",
        }
        ax.bar(x, values, color=[palette.get(solver, "#666666") for solver in solvers])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(solvers, rotation=20)
        for idx, value in enumerate(values):
            if not math.isnan(value):
                ax.text(idx, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    fig.savefig(output_dir / "metrics_mean.png", dpi=220)
    plt.close(fig)


def trajectory_colors():
    return {
        "ROUTE": ("#808080", "--", 1.8),
        "LBFGS": ("#0073F0", "-", 2.6),
        "IGO": ("#FF7A00", "-", 2.6),
        "META": ("#4C9F70", "-", 2.6),
    }


def plot_trajectory_shape(plt, rows, output_dir, request_id):
    if not rows:
        return

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["series"]].append(
            (
                int(row["sample_index"]),
                to_float(row["x"]),
                to_float(row["y"]),
                to_float(row["z"]),
            )
        )

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = trajectory_colors()

    for series, samples in grouped.items():
        samples = sorted(samples, key=lambda item: item[0])
        xs = [sample[1] for sample in samples if not math.isnan(sample[1])]
        ys = [sample[2] for sample in samples if not math.isnan(sample[2])]
        if not xs or not ys:
            continue

        color, linestyle, width = colors.get(series, ("#444444", "-", 2.0))
        ax.plot(xs, ys, linestyle=linestyle, linewidth=width, color=color, label=series)
        if series != "ROUTE":
            ax.scatter(xs[0], ys[0], s=45, color=color)
            ax.scatter(xs[-1], ys[-1], s=45, color=color)

    if "ROUTE" in grouped and grouped["ROUTE"]:
        route = sorted(grouped["ROUTE"], key=lambda item: item[0])
        ax.scatter(route[0][1], route[0][2], s=90, color="#2ca02c", label="START")
        ax.scatter(route[-1][1], route[-1][2], s=90, color="#d62728", label="GOAL")

    ax.set_title(f"Trajectory Shape Comparison (Request {request_id})")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_dir / f"request_{request_id}_trajectory.png", dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    plt = setup_matplotlib(output_dir)

    metrics_rows = load_csv_rows(output_dir / "trajectory_metrics.csv")
    if metrics_rows:
        summary_rows = compute_solver_means(metrics_rows)
        write_solver_mean_csv(summary_rows, output_dir)
        plot_mean_metrics(plt, summary_rows, output_dir)

    trajectory_rows = load_csv_rows(output_dir / f"request_{args.request_id}_trajectory.csv")
    if trajectory_rows:
        plot_trajectory_shape(plt, trajectory_rows, output_dir, args.request_id)


if __name__ == "__main__":
    main()
