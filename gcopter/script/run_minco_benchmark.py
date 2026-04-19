#!/usr/bin/env python3

import argparse
import copy
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build and run the GCOPTER MINCO LBFGS vs IGO benchmark inside Docker."
    )
    parser.add_argument(
        "--host-workspace",
        default=None,
        help="Host workspace root. Defaults to the current GCOPTER workspace.",
    )
    parser.add_argument(
        "--container-name",
        default="ros1_noetic",
        help="Docker container name.",
    )
    parser.add_argument(
        "--container-workspace",
        default="~/ws/gcopter",
        help="Workspace path inside the container.",
    )
    parser.add_argument(
        "--config-template",
        default=None,
        help="Base benchmark config YAML on the host.",
    )
    parser.add_argument(
        "--execution-mode",
        default="auto",
        choices=["auto", "local", "docker"],
        help="`local` runs directly in the current shell, `docker` wraps commands with docker exec.",
    )
    parser.add_argument("--mode", default="all", choices=["generate", "run", "all"])
    parser.add_argument("--num-instances", type=int, default=None)
    parser.add_argument("--generator-seed", type=int, default=None)
    parser.add_argument("--max-evaluations", type=int, default=None)
    parser.add_argument("--max-wall-time", type=float, default=None)
    parser.add_argument("--lbfgs-max-iterations", type=int, default=None)
    parser.add_argument("--igo-population", type=int, default=None)
    parser.add_argument("--igo-max-iterations", type=int, default=None)
    parser.add_argument("--igo-seeds", default=None, help="Comma-separated IGO seeds.")
    parser.add_argument("--feasibility-tol", type=float, default=None)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    return parser.parse_args()


def discover_host_workspace(explicit_root):
    if explicit_root:
        return Path(explicit_root).expanduser().resolve()
    return Path(__file__).resolve().parents[4]


def load_yaml(path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path, data):
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def run_command(cmd, cwd=None):
    print(f"$ {cmd}")
    completed = subprocess.run(cmd, shell=True, cwd=cwd, executable="/bin/bash")
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {cmd}")


def detect_execution_mode(requested_mode):
    if requested_mode != "auto":
        return requested_mode

    if Path("/.dockerenv").exists() or shutil_which("docker") is None:
        return "local"

    return "docker"


def shutil_which(name):
    for path in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(path) / name
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
    return None


def wrap_command_for_mode(command, execution_mode, args):
    if execution_mode == "local":
        return command

    return (
        f"docker exec {shlex.quote(args.container_name)} bash -lc "
        f"{shlex.quote(f'cd {args.container_workspace} && {command}')}"
    )


def main():
    args = parse_args()
    execution_mode = detect_execution_mode(args.execution_mode)

    host_workspace = discover_host_workspace(args.host_workspace)
    default_template = host_workspace / "src" / "GCOPTER" / "gcopter" / "config" / "minco_benchmark.yaml"
    config_template = Path(args.config_template).expanduser().resolve() if args.config_template else default_template
    plot_script = host_workspace / "src" / "GCOPTER" / "gcopter" / "script" / "plot_minco_benchmark.py"

    template = load_yaml(config_template)
    config = copy.deepcopy(template)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_rel = Path("src") / "GCOPTER" / "gcopter" / "benchmark_runs" / timestamp
    host_run_dir = host_workspace / run_rel
    host_run_dir.mkdir(parents=True, exist_ok=True)

    config.setdefault("run", {})
    config["run"]["mode"] = args.mode
    config["run"]["output_dir"] = run_rel.as_posix()

    if args.num_instances is not None:
        config.setdefault("instances", {})
        config["instances"]["num_instances"] = args.num_instances
    if args.generator_seed is not None:
        config.setdefault("instances", {})
        config["instances"]["generator_seed"] = args.generator_seed
    if args.max_evaluations is not None:
        config.setdefault("benchmark", {})
        config["benchmark"].setdefault("lbfgs", {})
        config["benchmark"].setdefault("igo", {})
        config["benchmark"]["lbfgs"]["max_evaluations"] = args.max_evaluations
        config["benchmark"]["igo"]["max_evaluations"] = args.max_evaluations
    if args.max_wall_time is not None:
        config.setdefault("benchmark", {})
        config["benchmark"].setdefault("lbfgs", {})
        config["benchmark"].setdefault("igo", {})
        config["benchmark"]["lbfgs"]["max_wall_time"] = args.max_wall_time
        config["benchmark"]["igo"]["max_wall_time"] = args.max_wall_time
    if args.lbfgs_max_iterations is not None:
        config.setdefault("benchmark", {})
        config["benchmark"].setdefault("lbfgs", {})
        config["benchmark"]["lbfgs"]["max_iterations"] = args.lbfgs_max_iterations
    if args.igo_population is not None:
        config.setdefault("benchmark", {})
        config["benchmark"].setdefault("igo", {})
        config["benchmark"]["igo"]["population"] = args.igo_population
    if args.igo_max_iterations is not None:
        config.setdefault("benchmark", {})
        config["benchmark"].setdefault("igo", {})
        config["benchmark"]["igo"]["max_iterations"] = args.igo_max_iterations
    if args.igo_seeds is not None:
        seeds = [int(item) for item in args.igo_seeds.split(",") if item.strip()]
        config.setdefault("benchmark", {})
        config["benchmark"].setdefault("igo", {})
        config["benchmark"]["igo"]["seeds"] = seeds
    if args.feasibility_tol is not None:
        config.setdefault("benchmark", {})
        config["benchmark"]["feasibility_tolerance"] = args.feasibility_tol

    host_config_path = host_run_dir / "benchmark_config.yaml"
    write_yaml(host_config_path, config)

    if execution_mode == "docker":
        runtime_config_path = f"{args.container_workspace.rstrip('/')}/{run_rel.as_posix()}/benchmark_config.yaml"
    else:
        runtime_config_path = str(host_config_path)

    if not args.skip_build:
        build_cmd = wrap_command_for_mode("catkin_make --pkg gcopter", execution_mode, args)
        run_command(build_cmd, cwd=str(host_workspace) if execution_mode == "local" else None)

    if not args.skip_run:
        runtime_cmd = (
            f"source /opt/ros/noetic/setup.bash && "
            f"source devel/setup.bash && "
            f"rosrun gcopter minco_benchmark --config {runtime_config_path}"
        )
        bench_cmd = wrap_command_for_mode(runtime_cmd, execution_mode, args)
        run_command(bench_cmd, cwd=str(host_workspace) if execution_mode == "local" else None)

    if not args.skip_plot:
        plot_cmd = f"{shlex.quote(sys.executable)} {shlex.quote(str(plot_script))} {shlex.quote(str(host_run_dir))}"
        run_command(plot_cmd, cwd=str(host_workspace))

    print(host_run_dir)


if __name__ == "__main__":
    main()
