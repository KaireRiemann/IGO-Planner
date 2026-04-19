#!/usr/bin/env python3

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize and plot paired LBFGS vs IGO benchmark results."
    )
    parser.add_argument("run_dir", help="Benchmark run directory containing raw_results.csv.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for plots and summary CSVs. Defaults to <run_dir>/plots.",
    )
    return parser.parse_args()


def to_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    if math.isfinite(value):
        return value
    return math.nan


def to_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def clean(values):
    return [v for v in values if not math.isnan(v)]


def median_iqr(values):
    values = clean(values)
    if not values:
        return math.nan, math.nan, math.nan
    arr = np.asarray(values, dtype=float)
    return float(np.median(arr)), float(np.percentile(arr, 25)), float(np.percentile(arr, 75))


def bootstrap_median_ci(differences, num_bootstrap=10000, seed=0):
    if len(differences) == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    diffs = np.asarray(differences, dtype=float)
    medians = np.empty(num_bootstrap, dtype=float)
    for i in range(num_bootstrap):
        sample = rng.choice(diffs, size=len(diffs), replace=True)
        medians[i] = np.median(sample)
    return float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def load_rows(csv_path):
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    parsed = []
    for row in rows:
        parsed.append(
            {
                "instance_id": to_int(row["instance_id"]),
                "solver": row["solver"],
                "seed": to_int(row["seed"]),
                "success": to_int(row["success"]),
                "has_solution": to_int(row["has_solution"]),
                "converged": to_int(row["converged"]),
                "hit_eval_budget": to_int(row["hit_eval_budget"]),
                "hit_time_budget": to_int(row["hit_time_budget"]),
                "solver_status": to_int(row["solver_status"]),
                "iterations": to_int(row["iterations"]),
                "eval_count": to_float(row["eval_count"]),
                "wall_time_sec": to_float(row["wall_time_sec"]),
                "objective": to_float(row["objective"]),
                "total_duration_sec": to_float(row["total_duration_sec"]),
                "trajectory_length": to_float(row["trajectory_length"]),
                "max_violation": to_float(row["max_violation"]),
                "status": row["status"],
            }
        )
    return parsed


def aggregate_by_instance(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["instance_id"]].append(row)

    paired_rows = []
    for instance_id in sorted(grouped):
        lbfgs_rows = [row for row in grouped[instance_id] if row["solver"] == "LBFGS"]
        igo_rows = [row for row in grouped[instance_id] if row["solver"] == "IGO"]
        if not lbfgs_rows or not igo_rows:
            continue

        lbfgs = lbfgs_rows[0]
        metrics = [
            "objective",
            "wall_time_sec",
            "trajectory_length",
            "max_violation",
            "total_duration_sec",
            "eval_count",
        ]

        paired = {
            "instance_id": instance_id,
            "lbfgs_success": lbfgs["success"],
            "lbfgs_objective": lbfgs["objective"],
            "lbfgs_wall_time_sec": lbfgs["wall_time_sec"],
            "lbfgs_trajectory_length": lbfgs["trajectory_length"],
            "lbfgs_max_violation": lbfgs["max_violation"],
            "lbfgs_total_duration_sec": lbfgs["total_duration_sec"],
            "lbfgs_eval_count": lbfgs["eval_count"],
            "igo_seed_count": len(igo_rows),
            "igo_success_rate": float(np.mean([row["success"] for row in igo_rows])),
            "igo_any_success": int(any(row["success"] for row in igo_rows)),
        }

        for metric in metrics:
            values = [row[metric] for row in igo_rows]
            med, q1, q3 = median_iqr(values)
            paired[f"igo_median_{metric}"] = med
            paired[f"igo_q1_{metric}"] = q1
            paired[f"igo_q3_{metric}"] = q3

        paired_rows.append(paired)

    return paired_rows


def write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_solver_summary(paired_rows):
    metrics = [
        ("objective", "lbfgs_objective", "igo_median_objective"),
        ("wall_time_sec", "lbfgs_wall_time_sec", "igo_median_wall_time_sec"),
        ("trajectory_length", "lbfgs_trajectory_length", "igo_median_trajectory_length"),
        ("max_violation", "lbfgs_max_violation", "igo_median_max_violation"),
        ("total_duration_sec", "lbfgs_total_duration_sec", "igo_median_total_duration_sec"),
        ("eval_count", "lbfgs_eval_count", "igo_median_eval_count"),
    ]

    summary_rows = []
    for label, lbfgs_key, igo_key in metrics:
        lbfgs_values = [row[lbfgs_key] for row in paired_rows]
        igo_values = [row[igo_key] for row in paired_rows]
        lb_med, lb_q1, lb_q3 = median_iqr(lbfgs_values)
        igo_med, igo_q1, igo_q3 = median_iqr(igo_values)
        summary_rows.append(
            {
                "metric": label,
                "lbfgs_median": lb_med,
                "lbfgs_q1": lb_q1,
                "lbfgs_q3": lb_q3,
                "igo_median": igo_med,
                "igo_q1": igo_q1,
                "igo_q3": igo_q3,
            }
        )

    success_rows = [
        {
            "metric": "success_rate",
            "lbfgs_median": float(np.mean([row["lbfgs_success"] for row in paired_rows])) if paired_rows else math.nan,
            "lbfgs_q1": math.nan,
            "lbfgs_q3": math.nan,
            "igo_median": float(np.mean([row["igo_success_rate"] for row in paired_rows])) if paired_rows else math.nan,
            "igo_q1": math.nan,
            "igo_q3": math.nan,
        },
        {
            "metric": "igo_any_success_rate",
            "lbfgs_median": math.nan,
            "lbfgs_q1": math.nan,
            "lbfgs_q3": math.nan,
            "igo_median": float(np.mean([row["igo_any_success"] for row in paired_rows])) if paired_rows else math.nan,
            "igo_q1": math.nan,
            "igo_q3": math.nan,
        },
    ]
    summary_rows.extend(success_rows)
    return summary_rows


def build_paired_stats(paired_rows):
    metrics = [
        ("objective", "lbfgs_objective", "igo_median_objective"),
        ("wall_time_sec", "lbfgs_wall_time_sec", "igo_median_wall_time_sec"),
        ("trajectory_length", "lbfgs_trajectory_length", "igo_median_trajectory_length"),
        ("max_violation", "lbfgs_max_violation", "igo_median_max_violation"),
        ("total_duration_sec", "lbfgs_total_duration_sec", "igo_median_total_duration_sec"),
        ("eval_count", "lbfgs_eval_count", "igo_median_eval_count"),
    ]

    stats_rows = []
    for label, lbfgs_key, igo_key in metrics:
        pairs = [
            (row[lbfgs_key], row[igo_key])
            for row in paired_rows
            if not math.isnan(row[lbfgs_key]) and not math.isnan(row[igo_key])
        ]
        diffs = [igo - lbfgs for lbfgs, igo in pairs]

        if pairs:
            try:
                wilcoxon_res = wilcoxon(
                    [pair[0] for pair in pairs],
                    [pair[1] for pair in pairs],
                    zero_method="pratt",
                    alternative="two-sided",
                )
                p_value = float(wilcoxon_res.pvalue)
            except ValueError:
                p_value = math.nan
            ci_low, ci_high = bootstrap_median_ci(diffs)
            median_diff = float(np.median(diffs))
        else:
            p_value = math.nan
            ci_low, ci_high, median_diff = math.nan, math.nan, math.nan

        stats_rows.append(
            {
                "metric": label,
                "n_pairs": len(pairs),
                "median_diff_igo_minus_lbfgs": median_diff,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "wilcoxon_pvalue": p_value,
            }
        )

    return stats_rows


def setup_matplotlib(output_dir):
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_success_rates(paired_rows, output_dir):
    plt = setup_matplotlib(output_dir)
    labels = ["LBFGS", "IGO mean seed", "IGO any seed"]
    values = [
        float(np.mean([row["lbfgs_success"] for row in paired_rows])) if paired_rows else 0.0,
        float(np.mean([row["igo_success_rate"] for row in paired_rows])) if paired_rows else 0.0,
        float(np.mean([row["igo_any_success"] for row in paired_rows])) if paired_rows else 0.0,
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values, color=["#0072b2", "#009e73", "#d55e00"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("rate")
    ax.set_title("Success Rates")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_dir / "success_rates.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_paired_objective(paired_rows, output_dir):
    plt = setup_matplotlib(output_dir)
    pairs = [
        (row["lbfgs_objective"], row["igo_median_objective"])
        for row in paired_rows
        if not math.isnan(row["lbfgs_objective"]) and not math.isnan(row["igo_median_objective"])
    ]
    if not pairs:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for lbfgs_obj, igo_obj in pairs:
        color = "#009e73" if igo_obj <= lbfgs_obj else "#d55e00"
        ax.plot([0, 1], [lbfgs_obj, igo_obj], color=color, alpha=0.45, linewidth=1.2)

    lbfgs_values = np.asarray([pair[0] for pair in pairs], dtype=float)
    igo_values = np.asarray([pair[1] for pair in pairs], dtype=float)
    ax.scatter(np.zeros_like(lbfgs_values), lbfgs_values, color="#0072b2", s=25, label="LBFGS")
    ax.scatter(np.ones_like(igo_values), igo_values, color="#009e73", s=25, label="IGO median")
    ax.scatter([0, 1], [np.median(lbfgs_values), np.median(igo_values)],
               color="black", s=80, marker="D", label="Median")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["LBFGS", "IGO median"])
    ax.set_ylabel("objective")
    ax.set_title("Paired Objective Comparison")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    if np.all(lbfgs_values > 0.0) and np.all(igo_values > 0.0):
        ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "paired_objective.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_difference_boxplots(paired_rows, output_dir):
    plt = setup_matplotlib(output_dir)
    diff_specs = [
        ("objective", "lbfgs_objective", "igo_median_objective"),
        ("wall time", "lbfgs_wall_time_sec", "igo_median_wall_time_sec"),
        ("length", "lbfgs_trajectory_length", "igo_median_trajectory_length"),
        ("max violation", "lbfgs_max_violation", "igo_median_max_violation"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    for ax, (title, lbfgs_key, igo_key) in zip(axes, diff_specs):
        diffs = [
            row[igo_key] - row[lbfgs_key]
            for row in paired_rows
            if not math.isnan(row[lbfgs_key]) and not math.isnan(row[igo_key])
        ]
        ax.boxplot([diffs] if diffs else [[]], labels=["IGO - LBFGS"], patch_artist=True)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_dir / "paired_differences.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else run_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_results = run_dir / "raw_results.csv"
    if not raw_results.exists():
        raise FileNotFoundError(f"Missing raw_results.csv: {raw_results}")

    rows = load_rows(raw_results)
    paired_rows = aggregate_by_instance(rows)

    paired_fields = [
        "instance_id",
        "lbfgs_success",
        "lbfgs_objective",
        "lbfgs_wall_time_sec",
        "lbfgs_trajectory_length",
        "lbfgs_max_violation",
        "lbfgs_total_duration_sec",
        "lbfgs_eval_count",
        "igo_seed_count",
        "igo_success_rate",
        "igo_any_success",
        "igo_median_objective",
        "igo_q1_objective",
        "igo_q3_objective",
        "igo_median_wall_time_sec",
        "igo_q1_wall_time_sec",
        "igo_q3_wall_time_sec",
        "igo_median_trajectory_length",
        "igo_q1_trajectory_length",
        "igo_q3_trajectory_length",
        "igo_median_max_violation",
        "igo_q1_max_violation",
        "igo_q3_max_violation",
        "igo_median_total_duration_sec",
        "igo_q1_total_duration_sec",
        "igo_q3_total_duration_sec",
        "igo_median_eval_count",
        "igo_q1_eval_count",
        "igo_q3_eval_count",
    ]
    write_csv(output_dir / "paired_instance_summary.csv", paired_rows, paired_fields)

    solver_summary = build_solver_summary(paired_rows)
    write_csv(
        output_dir / "solver_summary.csv",
        solver_summary,
        ["metric", "lbfgs_median", "lbfgs_q1", "lbfgs_q3", "igo_median", "igo_q1", "igo_q3"],
    )

    paired_stats = build_paired_stats(paired_rows)
    write_csv(
        output_dir / "paired_statistics.csv",
        paired_stats,
        ["metric", "n_pairs", "median_diff_igo_minus_lbfgs", "bootstrap_ci_low", "bootstrap_ci_high", "wilcoxon_pvalue"],
    )

    plot_success_rates(paired_rows, output_dir)
    plot_paired_objective(paired_rows, output_dir)
    plot_difference_boxplots(paired_rows, output_dir)


if __name__ == "__main__":
    main()
