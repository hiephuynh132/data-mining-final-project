#!/usr/bin/env python3
"""Export a V4 result directory to summary.md.

Usage:
  python export_summay.py /path/to/v4/results/20260604_234712
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path


ALGO_ORDER = ["kmeans", "diskmeans", "fdkm", "ifdkm", "ifdfd"]
PLUS_MINUS = chr(177)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export summary.csv/raw_run_log.csv in a V4 result folder to summary.md."
    )
    parser.add_argument(
        "result_dir",
        help="Result directory, e.g. /mnt/data-hdd/hiephd/kmean/v4/results/20260604_234712",
    )
    parser.add_argument(
        "--output",
        default="summary.md",
        help="Output markdown filename or path. Default: summary.md inside result_dir.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def mean_std(values: list[float]) -> tuple[float, float]:
    values = [v for v in values if not math.isnan(v)]
    if not values:
        return math.nan, math.nan
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(var)


def fmt_pair(mean: float, std: float) -> str:
    if math.isnan(mean):
        return "-"
    if math.isnan(std):
        std = 0.0
    return f"{mean:.2f}{PLUS_MINUS}{std:.2f}"


def fmt_gamma(value) -> str:
    if value in ("", "N/A", None):
        return "-"
    x = to_float(value)
    if math.isnan(x):
        return str(value)
    if x == 0:
        return "0"
    if abs(x) >= 1e4 or abs(x) < 1e-3:
        return f"{x:.0e}".replace("e+0", "e+").replace("e-0", "e-")
    if x.is_integer():
        return str(int(x))
    return f"{x:g}"


def result_timestamp_text(result_dir: Path) -> str:
    try:
        dt = datetime.strptime(result_dir.name, "%Y%m%d_%H%M%S")
    except ValueError:
        return result_dir.name
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def raw_log_path(result_dir: Path) -> Path | None:
    for name in ("raw_run_log.csv", "raw_runs.csv"):
        path = result_dir / name
        if path.exists():
            return path
    return None


def load_runtime_stats(result_dir: Path) -> tuple[dict, int | str]:
    path = raw_log_path(result_dir)
    if path is None:
        return {}, "?"

    rows = read_csv(path)
    runtimes = defaultdict(list)
    max_run = 0
    for row in rows:
        key = (row.get("dataset"), row.get("norm_mode"), row.get("algorithm"))
        runtimes[key].append(to_float(row.get("runtime_s")))
        run = to_float(row.get("run"))
        if not math.isnan(run):
            max_run = max(max_run, int(run))

    return {key: mean_std(vals) for key, vals in runtimes.items()}, max_run or "?"


def comparison_section(result_dir: Path) -> list[str]:
    path = result_dir / "comparison_to_paper.csv"
    if not path.exists():
        return []

    rows = read_csv(path)
    if not rows:
        return []

    mae_fields = [field for field in rows[0] if field.startswith("mae_")]
    header = ["Rank", "Mode", "Total MAE"] + [
        "MAE[" + field.removeprefix("mae_") + "]" for field in mae_fields
    ]

    lines = [
        "## Centering Modes Ranked vs Paper",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows:
        values = [
            row.get("rank", ""),
            row.get("norm_mode", ""),
            row.get("total_mae_vs_paper", row.get("total_mae", "")),
        ]
        values.extend(row.get(field, "") for field in mae_fields)
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    return lines


def build_summary_md(result_dir: Path) -> str:
    summary_csv = result_dir / "summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing summary.csv: {summary_csv}")

    rows = read_csv(summary_csv)
    runtime_stats, n_runs = load_runtime_stats(result_dir)

    datasets = list(dict.fromkeys(row["dataset"] for row in rows))
    norm_modes = list(dict.fromkeys(row["norm_mode"] for row in rows))
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["norm_mode"])].append(row)

    lines = [
        "# V4 Centering Study Results",
        "",
        f"_Generated from: `{result_dir.name}`_",
        f"_Timestamp: {result_timestamp_text(result_dir)}_",
        f"_Runs per combination: {n_runs}_",
        "",
    ]
    lines.extend(comparison_section(result_dir))
    lines.extend(["## Detailed Results", ""])

    for dataset in datasets:
        lines.extend([f"## {dataset}", ""])
        for mode in norm_modes:
            mode_rows = grouped.get((dataset, mode), [])
            if not mode_rows:
                continue
            mode_rows.sort(
                key=lambda row: (
                    ALGO_ORDER.index(row["algorithm"])
                    if row["algorithm"] in ALGO_ORDER
                    else len(ALGO_ORDER)
                )
            )

            lines.extend(
                [
                    f"### normalize: `{mode}`",
                    "",
                    "| Method | ACC (%) | NMI (%) | ARI (%) | gamma | Time (s) |",
                    "|--------|---------|---------|---------|-------|----------|",
                ]
            )
            for row in mode_rows:
                key = (row["dataset"], row["norm_mode"], row["algorithm"])
                time_mean, time_std = runtime_stats.get(key, (math.nan, math.nan))
                lines.append(
                    "| {method} | {acc} | {nmi} | {ari} | {gamma} | {time} |".format(
                        method=row["algorithm"].upper(),
                        acc=fmt_pair(to_float(row["acc_mean"]), to_float(row["acc_std"])),
                        nmi=fmt_pair(to_float(row["nmi_mean"]), to_float(row["nmi_std"])),
                        ari=fmt_pair(to_float(row["ari_mean"]), to_float(row["ari_std"])),
                        gamma=fmt_gamma(row.get("gamma")),
                        time=fmt_pair(time_mean, time_std),
                    )
                )
            lines.append("")

    return "\n".join(lines)


def output_path(result_dir: Path, output_arg: str) -> Path:
    path = Path(output_arg)
    if path.is_absolute():
        return path
    return result_dir / path


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir).expanduser().resolve()
    if not result_dir.is_dir():
        raise NotADirectoryError(f"Result directory not found: {result_dir}")

    md = build_summary_md(result_dir)
    out_path = output_path(result_dir, args.output)
    out_path.write_text(md, encoding="utf-8")
    print(f"summary.md written -> {out_path}")


if __name__ == "__main__":
    main()
