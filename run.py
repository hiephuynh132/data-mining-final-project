"""
V4 Centering Study — Entry Point

Runs all centering modes × algorithms × datasets, then ranks
each mode by closeness to the paper's reported ACC numbers.

Usage:
  python run.py                              # full study (config.yaml)
  python run.py --modes center standardize   # only specific modes
  python run.py --algorithms kmeans fdkm     # only specific algorithms
  python run.py --datasets glass ecoli       # only specific datasets
  python run.py --n-runs 3 --n-init 1        # faster (fewer runs)
"""

import argparse
import sys
from pathlib import Path

import yaml


def parse_args():
    p = argparse.ArgumentParser(description="V4 centering study")
    p.add_argument("--config",     default="config.yaml")
    p.add_argument("--datasets",   nargs="+", default=None,
                   help="Dataset keys (e.g. glass ecoli breast_cancer)")
    p.add_argument("--algorithms", nargs="+", default=None,
                   help="Algorithms (kmeans diskmeans fdkm ifdkm ifdfd)")
    p.add_argument("--modes",      nargs="+", default=None,
                   help="Centering modes (e.g. center standardize pca_whiten ...)")
    p.add_argument("--n-runs",     type=int, default=None)
    p.add_argument("--n-init",     type=int, default=None)
    p.add_argument("--seed",       type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)

    config_dir = str(config_path.parent)
    sys.path.insert(0, config_dir)

    # ── CLI overrides ──────────────────────────────────────────────────────
    if args.datasets:
        for k in cfg["datasets"]:
            cfg["datasets"][k]["enabled"] = (
                k in args.datasets
                or cfg["datasets"][k].get("name", "") in args.datasets
            )

    if args.algorithms:
        cfg["algorithms"] = args.algorithms

    if args.modes:
        for k in cfg["datasets"]:
            if cfg["datasets"][k].get("enabled", True):
                cfg["datasets"][k]["normalize_modes"] = args.modes

    if args.n_runs is not None:
        cfg["settings"]["n_runs"] = args.n_runs
    if args.n_init is not None:
        cfg["settings"]["n_init_per_run"] = args.n_init
    if args.seed is not None:
        cfg["settings"]["seed_base"] = args.seed

    # ── Run ─────────────────────────────────────────────────────────────────
    from src.runner import Runner, build_comparison

    runner     = Runner(cfg, config_dir)
    summaries  = runner.run_all()

    # ── Comparison table ─────────────────────────────────────────────────────
    build_comparison(
        summaries,
        paper_targets=cfg.get("paper_targets", {}),
        output_dir=runner.output_dir,
        n_runs=cfg["settings"]["n_runs"],
    )

    print(f"\nAll results -> {runner.output_dir}")


if __name__ == "__main__":
    main()
