#!/usr/bin/env python3

import argparse
import csv
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, median

import yaml


def parse_bool_flag(parser, name, default, help_text):
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=name.replace("-", "_"), action="store_true", help=help_text)
    group.add_argument(f"--no-{name}", dest=name.replace("-", "_"), action="store_false")
    parser.set_defaults(**{name.replace("-", "_"): default})


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run 100 random-map GCOPTER benchmark cases including META black-box optimizer."
    )
    parser.add_argument("--num-tests", type=int, default=100)
    parser.add_argument("--generator-seed", type=int, default=20260421)
    parser.add_argument("--config-template", default=None)
    parser.add_argument("--host-workspace", default=None)
    parser.add_argument("--container-name", default="ros1_noetic")
    parser.add_argument("--container-workspace", default="~/ws/gcopter")
    parser.add_argument("--execution-mode", choices=["auto", "local", "docker"], default="auto")
    parser.add_argument("--meta-seeds", default="0,1")
    parser.add_argument("--igo-seeds", default="0,1,2,3,4")
    parser.add_argument("--meta-population", type=int, default=None)
    parser.add_argument("--meta-max-iterations", type=int, default=None)
    parser.add_argument("--meta-max-evaluations", type=int, default=None)
    parser.add_argument("--meta-max-wall-time", type=float, default=None)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    parse_bool_flag(parser, "lbfgs", True, "Run LBFGS baseline.")
    parse_bool_flag(parser, "igo", True, "Run IGO baseline.")
    parse_bool_flag(parser, "meta", True, "Run META black-box optimizer.")
    parse_bool_flag(parser, "meta-pt", True, "Run direct P,T META variant.")
    return parser.parse_args()


def discover_host_workspace(explicit_root):
    if explicit_root:
        return Path(explicit_root).expanduser().resolve()
    return Path(__file__).resolve().parents[4]


def detect_execution_mode(requested_mode):
    if requested_mode != "auto":
        return requested_mode
    if Path("/.dockerenv").exists() or shutil_which("docker") is None:
        return "local"
    return "docker"


def shutil_which(name):
    for item in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(item) / name
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
    return None


def load_yaml(path):
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def write_yaml(path, data):
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream, sort_keys=False)


def parse_int_list(value):
    return [int(item) for item in value.split(",") if item.strip()]


def run_command(command, cwd=None):
    print(f"$ {command}")
    completed = subprocess.run(command, shell=True, cwd=cwd, executable="/bin/bash")
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {command}")


def wrap_command(command, mode, args):
    if mode == "local":
        return command
    container_command = f"cd {args.container_workspace} && {command}"
    return (
        f"docker exec {shlex.quote(args.container_name)} bash -lc "
        f"{shlex.quote(container_command)}"
    )


def to_float(row, key):
    try:
        return float(row.get(key, "nan"))
    except ValueError:
        return float("nan")


def finite_values(rows, key):
    values = []
    for row in rows:
        value = to_float(row, key)
        if value == value:
            values.append(value)
    return values


def summarize_results(raw_results_path, summary_path):
    with raw_results_path.open("r", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    grouped = {}
    for row in rows:
        grouped.setdefault(row["solver"], []).append(row)

    fields = [
        "solver",
        "rows",
        "success_rate",
        "has_solution_rate",
        "mean_wall_time_sec",
        "median_wall_time_sec",
        "mean_total_duration_sec",
        "mean_trajectory_length",
        "mean_max_violation",
        "mean_max_corridor_violation",
        "mean_max_velocity_violation",
        "mean_max_acceleration_violation",
    ]

    summaries = []
    for solver in sorted(grouped):
        solver_rows = grouped[solver]
        wall_times = finite_values(solver_rows, "wall_time_sec")
        durations = finite_values(solver_rows, "total_duration_sec")
        lengths = finite_values(solver_rows, "trajectory_length")
        max_violations = finite_values(solver_rows, "max_violation")
        corridor_violations = finite_values(solver_rows, "max_corridor_violation")
        velocity_violations = finite_values(solver_rows, "max_velocity_violation")
        acceleration_violations = finite_values(solver_rows, "max_acceleration_violation")
        success_count = sum(int(row.get("success", "0")) for row in solver_rows)
        solution_count = sum(int(row.get("has_solution", "0")) for row in solver_rows)
        summaries.append(
            {
                "solver": solver,
                "rows": len(solver_rows),
                "success_rate": success_count / max(1, len(solver_rows)),
                "has_solution_rate": solution_count / max(1, len(solver_rows)),
                "mean_wall_time_sec": mean(wall_times) if wall_times else float("nan"),
                "median_wall_time_sec": median(wall_times) if wall_times else float("nan"),
                "mean_total_duration_sec": mean(durations) if durations else float("nan"),
                "mean_trajectory_length": mean(lengths) if lengths else float("nan"),
                "mean_max_violation": mean(max_violations) if max_violations else float("nan"),
                "mean_max_corridor_violation": mean(corridor_violations) if corridor_violations else float("nan"),
                "mean_max_velocity_violation": mean(velocity_violations) if velocity_violations else float("nan"),
                "mean_max_acceleration_violation": mean(acceleration_violations)
                if acceleration_violations
                else float("nan"),
            }
        )

    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)

    for item in summaries:
        print(
            "{solver}: success={success_rate:.3f}, wall_mean={mean_wall_time_sec:.4f}s, "
            "duration_mean={mean_total_duration_sec:.3f}, max_violation_mean={mean_max_violation:.6f}".format(
                **item
            )
        )


def main():
    args = parse_args()
    host_workspace = discover_host_workspace(args.host_workspace)
    package_root = host_workspace / "src" / "IGO-Planner" / "gcopter"
    template_path = (
        Path(args.config_template).expanduser().resolve()
        if args.config_template
        else package_root / "config" / "minco_benchmark.yaml"
    )
    execution_mode = detect_execution_mode(args.execution_mode)

    config = load_yaml(template_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_rel = Path("src") / "IGO-Planner" / "gcopter" / "benchmark_runs" / timestamp
    host_run_dir = host_workspace / run_rel
    host_run_dir.mkdir(parents=True, exist_ok=True)

    config.setdefault("run", {})
    config["run"]["mode"] = "all"
    config["run"]["output_dir"] = run_rel.as_posix()
    config.setdefault("instances", {})
    config["instances"]["num_instances"] = args.num_tests
    config["instances"]["generator_seed"] = args.generator_seed
    config.setdefault("benchmark", {})
    config["benchmark"]["run_lbfgs"] = args.lbfgs
    config["benchmark"]["run_igo"] = args.igo
    config["benchmark"].setdefault("igo", {})
    config["benchmark"]["igo"]["seeds"] = parse_int_list(args.igo_seeds)
    config["benchmark"].setdefault("meta", {})
    config["benchmark"]["meta"]["enabled"] = args.meta
    config["benchmark"]["meta"]["run_pt"] = args.meta_pt
    config["benchmark"]["meta"]["seeds"] = parse_int_list(args.meta_seeds)

    if args.meta_population is not None:
        config["benchmark"]["meta"]["population"] = args.meta_population
    if args.meta_max_iterations is not None:
        config["benchmark"]["meta"]["max_iterations"] = args.meta_max_iterations
    if args.meta_max_evaluations is not None:
        config["benchmark"]["meta"]["max_evaluations"] = args.meta_max_evaluations
    if args.meta_max_wall_time is not None:
        config["benchmark"]["meta"]["max_wall_time"] = args.meta_max_wall_time

    config_path = host_run_dir / "benchmark_config.yaml"
    write_yaml(config_path, config)

    runtime_config = (
        f"{args.container_workspace.rstrip('/')}/{run_rel.as_posix()}/benchmark_config.yaml"
        if execution_mode == "docker"
        else str(config_path)
    )

    if not args.skip_build:
        build_command = wrap_command("catkin_make --pkg gcopter", execution_mode, args)
        run_command(build_command, cwd=str(host_workspace) if execution_mode == "local" else None)

    if not args.skip_run:
        run_inner = (
            "source /opt/ros/noetic/setup.bash && "
            "source devel/setup.bash && "
            f"rosrun gcopter minco_benchmark --config {shlex.quote(runtime_config)}"
        )
        run_command(wrap_command(run_inner, execution_mode, args),
                    cwd=str(host_workspace) if execution_mode == "local" else None)

    raw_results = host_run_dir / "raw_results.csv"
    if raw_results.exists():
        summarize_results(raw_results, host_run_dir / "solver_summary.csv")
    else:
        print(f"raw_results.csv not found yet: {raw_results}")

    print(host_run_dir)


if __name__ == "__main__":
    main()
