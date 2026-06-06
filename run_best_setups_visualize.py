#!/usr/bin/env python3
"""Run best_setups.csv configurations and visualize clustering results.

Example:
  python run_best_setups_visualize.py \
    results/20260604_234712/best_setups.csv \
    --config config.yaml
"""

from __future__ import annotations

import argparse
import csv
import gc
import html
import json
import re
import sys
import threading
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import psutil
import yaml
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import StandardScaler


ALGORITHM_ORDER = ["kmeans", "diskmeans", "fdkm", "ifdkm", "ifdfd"]
ALGORITHM_LABELS = {
    "kmeans": "KMeans",
    "diskmeans": "DisKmeans",
    "fdkm": "FDKM",
    "ifdkm": "IFDKM",
    "ifdfd": "IFDFD",
}
METRIC_COLORS = {"acc": "#2878B5", "nmi": "#E07A35", "ari": "#3A9D5D"}
RESULT_FIELDS = [
    "dataset", "algorithm", "norm_mode", "gamma", "seed",
    "acc", "nmi", "ari", "runtime_s", "n_iter",
    "memory_baseline_mb", "memory_peak_mb", "memory_delta_mb",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run setups from best_setups.csv and visualize cluster quality."
    )
    parser.add_argument(
        "best_setups",
        nargs="?",
        help="Path to best_setups.csv. Not required with --render-only.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.yaml. Default: config.yaml beside this script.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: <best_setups directory>/visualizations",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Only run selected dataset keys or names.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=ALGORITHM_ORDER,
        default=None,
        help="Only run selected algorithms.",
    )
    parser.add_argument(
        "--max-plot-points",
        type=int,
        default=5000,
        help="Maximum points drawn per dataset. Clustering still uses all loaded samples.",
    )
    parser.add_argument(
        "--n-init",
        type=int,
        default=None,
        help="Override n_init_per_run for kernel algorithms.",
    )
    parser.add_argument(
        "--render-only",
        default=None,
        metavar="DATA_FILE",
        help="Skip clustering and redraw plots from visualization_data.npz.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "dataset"


def parse_gamma(value: str) -> float | None:
    if value in ("", "N/A", None):
        return None
    return float(value)


def class_names_from_dataset(
    data_dir: str, dataset_config: dict, y: np.ndarray
) -> list[str]:
    n_classes = len(np.unique(y))
    dataset_name = dataset_config.get("name", "")
    data_path = Path(data_dir) / dataset_config.get("path", "")
    data_format = dataset_config.get("format", "")

    if data_format in ("sklearn_digits", "mnist"):
        return [f"Digit {idx}" for idx in range(n_classes)]
    if data_format == "image":
        return [f"Object {idx + 1}" for idx in range(n_classes)]
    if data_format == "covertype":
        cover_types = [
            "Spruce/Fir", "Lodgepole Pine", "Ponderosa Pine",
            "Cottonwood/Willow", "Aspen", "Douglas-fir", "Krummholz",
        ]
        return cover_types[:n_classes]

    raw_labels = []
    if data_format == "csv":
        with data_path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            raw_labels = sorted({row[dataset_config["label_col"]] for row in reader})
    elif data_format == "csv_noheader":
        with data_path.open(newline="", encoding="utf-8-sig") as f:
            rows = csv.reader(f)
            raw_labels = sorted(
                {row[int(dataset_config.get("label_col", -1))] for row in rows},
                key=lambda value: float(value),
            )
    elif data_format == "whitespace":
        with data_path.open(encoding="utf-8-sig") as f:
            raw_labels = sorted({
                line.split()[int(dataset_config.get("label_col", -1))]
                for line in f
                if line.strip()
            })
    elif data_format == "segmentation":
        skip_rows = int(dataset_config.get("skip_rows", 5))
        with data_path.open(newline="", encoding="utf-8-sig") as f:
            rows = csv.reader(f)
            for row_idx, row in enumerate(rows):
                if row_idx < skip_rows or not row:
                    continue
                if len(row) > 1:
                    raw_labels.append(row[0])
        raw_labels = sorted(set(raw_labels))

    if dataset_name == "Breast_Cancer":
        descriptions = {"B": "Benign", "M": "Malignant"}
        return [descriptions.get(value, value) for value in raw_labels]
    if dataset_name == "glass+identification":
        descriptions = {
            "1": "Building window (float)",
            "2": "Building window (non-float)",
            "3": "Vehicle window (float)",
            "4": "Vehicle window (non-float)",
            "5": "Container",
            "6": "Tableware",
            "7": "Headlamp",
        }
        return [descriptions.get(value, f"Glass type {value}") for value in raw_labels]
    if dataset_name == "ecoli":
        descriptions = {
            "cp": "Cytoplasm",
            "im": "Inner membrane",
            "pp": "Periplasm",
            "imU": "Inner membrane (uncleavable)",
            "om": "Outer membrane",
            "omL": "Outer membrane lipoprotein",
            "imL": "Inner membrane lipoprotein",
            "imS": "Inner membrane (cleavable)",
        }
        return [
            f"{value}: {descriptions[value]}" if value in descriptions else value
            for value in raw_labels
        ]
    if dataset_name == "image+segmentation":
        return [value.title() for value in raw_labels]
    if len(raw_labels) == n_classes:
        return [str(value) for value in raw_labels]
    return [f"Class {idx}" for idx in range(n_classes)]


def append_csv(path: Path, row: dict) -> None:
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})


def measure_peak_memory(function, poll_interval: float = 0.005):
    """Run function and sample process RSS until it completes."""
    gc.collect()
    process = psutil.Process()
    baseline = process.memory_info().rss
    peak = baseline
    stop_event = threading.Event()

    def sample_memory():
        nonlocal peak
        while not stop_event.is_set():
            peak = max(peak, process.memory_info().rss)
            stop_event.wait(poll_interval)

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    try:
        result = function()
    finally:
        peak = max(peak, process.memory_info().rss)
        stop_event.set()
        sampler.join()

    to_mb = 1024.0 * 1024.0
    return result, {
        "memory_baseline_mb": baseline / to_mb,
        "memory_peak_mb": peak / to_mb,
        "memory_delta_mb": max(0, peak - baseline) / to_mb,
    }


def save_visualization_data(
    path: Path,
    datasets: list[dict],
) -> None:
    metadata = {"datasets": []}
    arrays = {}
    for dataset_idx, item in enumerate(datasets):
        prefix = f"dataset_{dataset_idx}"
        results = item["results"]
        metadata["datasets"].append({
            "prefix": prefix,
            "dataset": item["dataset"],
            "class_names": item["class_names"],
            "algorithms": [algo for algo in ALGORITHM_ORDER if algo in results],
            "results": {
                algo: {
                    key: value
                    for key, value in result.items()
                    if key != "labels"
                }
                for algo, result in results.items()
            },
        })
        arrays[f"{prefix}_embedding"] = item["embedding"]
        arrays[f"{prefix}_plot_indices"] = item["plot_indices"]
        arrays[f"{prefix}_y_true"] = item["y_true"]
        for algorithm, result in results.items():
            arrays[f"{prefix}_labels_{algorithm}"] = result["labels"]
    arrays["metadata_json"] = np.asarray(json.dumps(metadata))
    temp_path = path.with_name(f"{path.stem}.tmp.npz")
    np.savez_compressed(temp_path, **arrays)
    temp_path.replace(path)


def load_visualization_data(path: Path) -> list[dict]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        datasets = []
        for item in metadata["datasets"]:
            prefix = item["prefix"]
            results = {}
            for algorithm in item["algorithms"]:
                result = item["results"][algorithm]
                result["labels"] = data[f"{prefix}_labels_{algorithm}"].copy()
                results[algorithm] = result
            datasets.append({
                "dataset": item["dataset"],
                "class_names": item.get(
                    "class_names",
                    [
                        f"Class {idx}"
                        for idx in range(
                            len(np.unique(data[f"{prefix}_y_true"]))
                        )
                    ],
                ),
                "embedding": data[f"{prefix}_embedding"].copy(),
                "plot_indices": data[f"{prefix}_plot_indices"].copy(),
                "y_true": data[f"{prefix}_y_true"].copy(),
                "results": results,
            })
        return datasets


def select_plot_indices(y: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    n = len(y)
    if max_points <= 0 or n <= max_points:
        return np.arange(n)

    rng = np.random.default_rng(seed)
    selected = []
    classes, counts = np.unique(y, return_counts=True)
    allocations = np.maximum(1, np.floor(counts / n * max_points).astype(int))

    while allocations.sum() > max_points:
        idx = int(np.argmax(allocations))
        if allocations[idx] > 1:
            allocations[idx] -= 1
        else:
            break
    while allocations.sum() < max_points:
        capacity = counts - allocations
        idx = int(np.argmax(capacity))
        if capacity[idx] <= 0:
            break
        allocations[idx] += 1

    for class_value, amount in zip(classes, allocations):
        class_indices = np.flatnonzero(y == class_value)
        selected.extend(rng.choice(class_indices, size=amount, replace=False))
    return np.sort(np.asarray(selected, dtype=int))


def shared_pca_embedding(
    X_raw: np.ndarray, y: np.ndarray, max_points: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    indices = select_plot_indices(y, max_points, seed)
    X_plot = StandardScaler().fit_transform(X_raw[indices])
    embedding = PCA(n_components=2, svd_solver="randomized", random_state=seed).fit_transform(
        X_plot
    )
    return embedding, indices


def align_labels(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    true_values = np.unique(y_true)
    pred_values = np.unique(y_pred)
    matrix = contingency_matrix(y_true, y_pred)
    true_idx, pred_idx = linear_sum_assignment(-matrix)
    mapping = {
        pred_values[pred_col]: true_values[true_row]
        for true_row, pred_col in zip(true_idx, pred_idx)
    }
    next_label = int(true_values.max()) + 1 if len(true_values) else 0
    aligned = np.empty_like(y_pred, dtype=int)
    for pred_value in pred_values:
        mapped = mapping.get(pred_value)
        if mapped is None:
            mapped = next_label
            next_label += 1
        aligned[y_pred == pred_value] = mapped
    return aligned


def categorical_colors(n_classes: int) -> tuple[ListedColormap, BoundaryNorm]:
    if n_classes <= 10:
        colors = plt.get_cmap("tab10").colors[:n_classes]
    elif n_classes <= 20:
        colors = plt.get_cmap("tab20").colors[:n_classes]
    else:
        colors = plt.get_cmap("hsv")(np.linspace(0, 1, n_classes, endpoint=False))
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, n_classes + 0.5), n_classes)
    return cmap, norm


def run_kernel(
    AlgoClass,
    X: np.ndarray,
    n_clusters: int,
    gamma: float,
    seed: int,
    n_init: int,
    max_iter: int,
    tol: float,
    make_random_init,
) -> tuple[np.ndarray, float, int]:
    best_labels = None
    best_objective = -np.inf
    best_runtime = 0.0
    best_n_iter = 0
    for init_idx in range(n_init):
        init_seed = seed + init_idx * 1000
        init_labels = make_random_init(len(X), n_clusters, init_seed)
        algorithm = AlgoClass(n_clusters=n_clusters, max_iter=max_iter, tol=tol)
        labels, runtime = algorithm.fit(X, gamma=gamma, init_labels=init_labels)
        objective = (
            algorithm.objective_
            if algorithm.objective_ is not None
            else -np.inf
        )
        if objective > best_objective:
            best_objective = objective
            best_labels = labels
            best_runtime = runtime
            best_n_iter = int(algorithm.n_iter_)
    return best_labels, best_runtime, best_n_iter


def plot_clusters(
    dataset: str,
    embedding: np.ndarray,
    plot_indices: np.ndarray,
    y_true: np.ndarray,
    results: dict[str, dict],
    class_names: list[str],
    output_path: Path,
) -> None:
    algorithms = [algo for algo in ALGORITHM_ORDER if algo in results]
    panels = [("Ground Truth", y_true, None)] + [
        (ALGORITHM_LABELS[algo], results[algo]["labels"], results[algo])
        for algo in algorithms
    ]
    n_cols = 3
    n_rows = int(np.ceil(len(panels) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5.2 * n_rows), squeeze=False)
    n_classes = len(np.unique(y_true))
    cmap, color_norm = categorical_colors(n_classes)

    for axis, (title, labels, result) in zip(axes.flat, panels):
        shown_labels = labels[plot_indices]
        axis.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=shown_labels,
            cmap=cmap,
            norm=color_norm,
            s=12,
            alpha=0.78,
            linewidths=0,
            rasterized=True,
        )
        for class_idx in np.unique(shown_labels):
            class_points = embedding[shown_labels == class_idx]
            if len(class_points) == 0:
                continue
            center = class_points.mean(axis=0)
            axis.scatter(
                center[0],
                center[1],
                marker="X",
                s=150,
                c=[cmap(color_norm(int(class_idx)))],
                edgecolors="#111111",
                linewidths=1.2,
                zorder=5,
            )
        if result is not None:
            wrong = shown_labels != y_true[plot_indices]
            if np.any(wrong):
                axis.scatter(
                    embedding[wrong, 0],
                    embedding[wrong, 1],
                    marker="o",
                    facecolors="none",
                    edgecolors="#111111",
                    s=18,
                    linewidths=0.5,
                    alpha=0.6,
                    rasterized=True,
                )
            title += (
                f"\nACC {result['acc']:.2f}% | "
                f"{result['norm_mode']} | gamma {result['gamma_text']}"
            )
        axis.set_title(title, fontsize=11)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)

    for axis in axes.flat[len(panels):]:
        axis.axis("off")

    if n_classes <= 20:
        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=7,
                markerfacecolor=cmap(color_norm(class_idx)),
                markeredgecolor="none",
                label=(
                    class_names[class_idx]
                    if class_idx < len(class_names)
                    else f"Class {class_idx}"
                ),
            )
            for class_idx in range(n_classes)
        ]
        fig.legend(
            handles=legend_handles + [
                Line2D(
                    [0],
                    [0],
                    marker="X",
                    linestyle="",
                    markersize=9,
                    markerfacecolor="#BBBBBB",
                    markeredgecolor="#111111",
                    label="Cluster center",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    markersize=7,
                    markerfacecolor="none",
                    markeredgecolor="#111111",
                    label="Incorrect point",
                ),
            ],
            loc="lower center",
            ncols=min(n_classes + 2, 10),
            frameon=False,
            title="Class labels",
        )
        bottom_margin = 0.08 if n_classes <= 10 else 0.12
    else:
        scalar_map = plt.cm.ScalarMappable(cmap=cmap, norm=color_norm)
        scalar_map.set_array([])
        colorbar = fig.colorbar(
            scalar_map,
            ax=axes.ravel().tolist(),
            orientation="horizontal",
            fraction=0.035,
            pad=0.06,
        )
        colorbar.set_label("Class label")
        tick_step = max(1, int(np.ceil(n_classes / 20)))
        colorbar.set_ticks(np.arange(0, n_classes, tick_step))
        fig.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    marker="X",
                    linestyle="",
                    markersize=9,
                    markerfacecolor="#BBBBBB",
                    markeredgecolor="#111111",
                    label="Cluster center",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    markersize=7,
                    markerfacecolor="none",
                    markeredgecolor="#111111",
                    label="Incorrect point",
                ),
            ],
            loc="lower center",
            ncols=2,
            frameon=False,
        )
        bottom_margin = 0.12

    fig.suptitle(
        f"{dataset}: shared PCA view (colored X = cluster center)",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, bottom_margin, 1, 0.96))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_metrics(dataset: str, results: dict[str, dict], output_path: Path) -> None:
    algorithms = [algo for algo in ALGORITHM_ORDER if algo in results]
    x = np.arange(len(algorithms))
    width = 0.24
    fig, axis = plt.subplots(figsize=(11, 5.5))

    for offset, metric in zip((-width, 0, width), ("acc", "nmi", "ari")):
        values = [results[algo][metric] for algo in algorithms]
        bars = axis.bar(
            x + offset,
            values,
            width,
            label=metric.upper(),
            color=METRIC_COLORS[metric],
        )
        axis.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)

    axis.set_title(f"{dataset}: clustering metrics", fontsize=14, fontweight="bold")
    axis.set_ylabel("Score (%)")
    axis.set_ylim(0, 105)
    axis.set_xticks(x, [ALGORITHM_LABELS[algo] for algo in algorithms])
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False, ncols=3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_runtime_iterations(
    dataset: str, results: dict[str, dict], output_path: Path
) -> None:
    algorithms = [algo for algo in ALGORITHM_ORDER if algo in results]
    x = np.arange(len(algorithms))
    runtimes = [float(results[algo]["runtime_s"]) for algo in algorithms]
    iterations = [float(results[algo].get("n_iter", np.nan)) for algo in algorithms]

    fig, runtime_axis = plt.subplots(figsize=(11, 5.5))
    bars = runtime_axis.bar(
        x,
        runtimes,
        width=0.58,
        color="#2878B5",
        alpha=0.86,
        label="Runtime",
    )
    runtime_axis.bar_label(
        bars,
        labels=[f"{value:.3f}s" for value in runtimes],
        padding=3,
        fontsize=8,
    )
    runtime_axis.set_ylabel("Runtime (seconds)", color="#2878B5")
    runtime_axis.tick_params(axis="y", labelcolor="#2878B5")
    runtime_axis.set_xticks(x, [ALGORITHM_LABELS[algo] for algo in algorithms])
    runtime_axis.grid(axis="y", alpha=0.2)

    iteration_axis = runtime_axis.twinx()
    valid = ~np.isnan(iterations)
    if np.any(valid):
        iteration_axis.plot(
            x[valid],
            np.asarray(iterations)[valid],
            color="#D9534F",
            marker="o",
            markersize=7,
            linewidth=2,
            label="Iterations",
        )
        for x_value, iteration in zip(x[valid], np.asarray(iterations)[valid]):
            iteration_axis.annotate(
                f"{int(iteration)}",
                (x_value, iteration),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color="#B23B37",
            )
    iteration_axis.set_ylabel("Iterations to converge", color="#D9534F")
    iteration_axis.tick_params(axis="y", labelcolor="#D9534F")
    iteration_axis.set_ylim(bottom=0)

    handles_1, labels_1 = runtime_axis.get_legend_handles_labels()
    handles_2, labels_2 = iteration_axis.get_legend_handles_labels()
    runtime_axis.legend(
        handles_1 + handles_2,
        labels_1 + labels_2,
        frameon=False,
        loc="upper left",
    )
    runtime_axis.set_title(
        f"{dataset}: runtime and convergence iterations",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_memory_usage(
    dataset: str, results: dict[str, dict], output_path: Path
) -> None:
    algorithms = [algo for algo in ALGORITHM_ORDER if algo in results]
    deltas = [
        float(results[algo].get("memory_delta_mb", np.nan))
        for algo in algorithms
    ]
    peaks = [
        float(results[algo].get("memory_peak_mb", np.nan))
        for algo in algorithms
    ]
    if not np.any(np.isfinite(deltas)):
        return

    x = np.arange(len(algorithms))
    fig, axis = plt.subplots(figsize=(11, 5.5))
    bars = axis.bar(x, deltas, width=0.58, color="#6A5ACD", alpha=0.88)
    labels = [
        f"+{delta:.1f} MB\npeak {peak:.1f} MB"
        if np.isfinite(delta) and np.isfinite(peak)
        else "-"
        for delta, peak in zip(deltas, peaks)
    ]
    axis.bar_label(bars, labels=labels, padding=4, fontsize=8)
    axis.set_xticks(x, [ALGORITHM_LABELS[algo] for algo in algorithms])
    axis.set_ylabel("Additional peak RSS (MB)")
    axis.set_title(
        f"{dataset}: peak memory usage by algorithm",
        fontsize=14,
        fontweight="bold",
    )
    axis.grid(axis="y", alpha=0.22)
    axis.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_centroid_distances(
    dataset: str,
    embedding: np.ndarray,
    plot_indices: np.ndarray,
    y_true: np.ndarray,
    results: dict[str, dict],
    class_names: list[str],
    output_path: Path,
) -> None:
    algorithms = [algo for algo in ALGORITHM_ORDER if algo in results]
    shown_true = y_true[plot_indices]
    classes = np.unique(shown_true).astype(int)
    true_centers = {}
    for class_idx in classes:
        points = embedding[shown_true == class_idx]
        if len(points):
            true_centers[class_idx] = points.mean(axis=0)

    distances = np.full((len(algorithms), len(classes)), np.nan)
    for row_idx, algorithm in enumerate(algorithms):
        shown_pred = results[algorithm]["labels"][plot_indices]
        for col_idx, class_idx in enumerate(classes):
            points = embedding[shown_pred == class_idx]
            if len(points) and class_idx in true_centers:
                predicted_center = points.mean(axis=0)
                distances[row_idx, col_idx] = np.linalg.norm(
                    predicted_center - true_centers[class_idx]
                )

    finite_values = distances[np.isfinite(distances)]
    vmax = float(np.max(finite_values)) if finite_values.size else 1.0
    vmax = max(vmax, 1e-12)
    fig_width = min(30, max(10, len(classes) * 0.72))
    fig_height = max(4.8, len(algorithms) * 0.75 + 2.2)
    fig, axis = plt.subplots(figsize=(fig_width, fig_height))
    image = axis.imshow(
        distances,
        cmap="YlOrRd",
        vmin=0,
        vmax=vmax,
        aspect="auto",
    )

    if len(classes) <= 25:
        for row_idx in range(len(algorithms)):
            for col_idx in range(len(classes)):
                value = distances[row_idx, col_idx]
                if np.isfinite(value):
                    axis.text(
                        col_idx,
                        row_idx,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="black" if value < vmax * 0.65 else "white",
                    )

    labels = [
        class_names[class_idx]
        if class_idx < len(class_names)
        else f"Class {class_idx}"
        for class_idx in classes
    ]
    if len(classes) <= 30:
        tick_indices = np.arange(len(classes))
    else:
        tick_step = max(1, int(np.ceil(len(classes) / 25)))
        tick_indices = np.arange(0, len(classes), tick_step)
    axis.set_xticks(tick_indices, [labels[idx] for idx in tick_indices])
    axis.tick_params(axis="x", labelrotation=45, labelsize=8)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
    axis.set_yticks(
        np.arange(len(algorithms)),
        [ALGORITHM_LABELS[algo] for algo in algorithms],
    )
    axis.set_xlabel("Class")
    axis.set_ylabel("Algorithm")
    axis.set_title(
        f"{dataset}: predicted centroid distance from ground truth",
        fontsize=14,
        fontweight="bold",
    )
    colorbar = fig.colorbar(image, ax=axis, fraction=0.03, pad=0.03)
    colorbar.set_label("Euclidean distance in shared PCA space (lower is better)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_ground_truth_with_predicted_centers(
    dataset: str,
    embedding: np.ndarray,
    plot_indices: np.ndarray,
    y_true: np.ndarray,
    results: dict[str, dict],
    class_names: list[str],
    output_path: Path,
) -> None:
    algorithms = [algo for algo in ALGORITHM_ORDER if algo in results]
    shown_true = y_true[plot_indices]
    n_classes = len(np.unique(y_true))
    cmap, color_norm = categorical_colors(n_classes)
    center_sources = [("Ground Truth", None)] + [
        (ALGORITHM_LABELS[algo], algo) for algo in algorithms
    ]
    center_markers = {
        "Ground Truth": "*",
        "KMeans": "X",
        "DisKmeans": "P",
        "FDKM": "D",
        "IFDKM": "^",
        "IFDFD": "s",
    }
    center_sizes = {
        "Ground Truth": 260,
        "KMeans": 175,
        "DisKmeans": 165,
        "FDKM": 150,
        "IFDKM": 165,
        "IFDFD": 145,
    }

    fig, axis = plt.subplots(figsize=(14, 9))
    axis.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=shown_true,
        cmap=cmap,
        norm=color_norm,
        s=14,
        alpha=0.58,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )

    true_centers = {}
    for class_idx in np.unique(shown_true):
        points = embedding[shown_true == class_idx]
        if len(points):
            true_centers[int(class_idx)] = points.mean(axis=0)

    for source_name, algorithm in center_sources:
        center_labels = (
            shown_true
            if algorithm is None
            else results[algorithm]["labels"][plot_indices]
        )
        for class_idx in np.unique(center_labels):
            class_points = embedding[center_labels == class_idx]
            if not len(class_points):
                continue
            center = class_points.mean(axis=0)
            axis.scatter(
                center[0],
                center[1],
                marker=center_markers[source_name],
                s=center_sizes[source_name],
                c=[cmap(color_norm(int(class_idx)))],
                edgecolors="#111111",
                linewidths=1.25,
                zorder=4 if algorithm is not None else 6,
            )
            if algorithm is not None and int(class_idx) in true_centers:
                ground_center = true_centers[int(class_idx)]
                axis.plot(
                    [ground_center[0], center[0]],
                    [ground_center[1], center[1]],
                    color=cmap(color_norm(int(class_idx))),
                    linewidth=0.8,
                    alpha=0.42,
                    zorder=2,
                )

            if algorithm is None and n_classes <= 20:
                short_name = (
                    class_names[int(class_idx)]
                    if int(class_idx) < len(class_names)
                    else f"Class {int(class_idx)}"
                )
                axis.annotate(
                    short_name,
                    center,
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=7,
                    color="#111111",
                    bbox={
                        "boxstyle": "round,pad=0.18",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.72,
                    },
                )

    class_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=7,
            markerfacecolor=cmap(color_norm(class_idx)),
            markeredgecolor="none",
            label=(
                class_names[class_idx]
                if class_idx < len(class_names)
                else f"Class {class_idx}"
            ),
        )
        for class_idx in range(n_classes)
    ]
    algorithm_handles = [
        Line2D(
            [0],
            [0],
            marker=center_markers[source_name],
            linestyle="",
            markersize=10 if source_name != "Ground Truth" else 12,
            markerfacecolor="#BBBBBB",
            markeredgecolor="#111111",
            label=source_name,
        )
        for source_name, _ in center_sources
    ]

    axis.set_xlabel("PCA component 1")
    axis.set_ylabel("PCA component 2")
    axis.grid(alpha=0.18)
    axis.set_title(
        f"{dataset}: ground truth data with all algorithm centroids",
        fontsize=15,
        fontweight="bold",
    )

    algorithm_legend = axis.legend(
        handles=algorithm_handles,
        loc="upper right",
        frameon=True,
        title="Centroid source",
    )
    axis.add_artist(algorithm_legend)
    if n_classes <= 20:
        axis.legend(
            handles=class_handles,
            loc="upper left",
            ncols=1 if n_classes <= 10 else 2,
            frameon=True,
            title="Ground-truth classes",
        )
    else:
        scalar_map = plt.cm.ScalarMappable(cmap=cmap, norm=color_norm)
        scalar_map.set_array([])
        colorbar = fig.colorbar(
            scalar_map,
            ax=axis,
            orientation="vertical",
            fraction=0.025,
            pad=0.03,
        )
        colorbar.set_label("Ground-truth class")
        tick_step = max(1, int(np.ceil(n_classes / 20)))
        colorbar.set_ticks(np.arange(0, n_classes, tick_step))

    metric_text = "\n".join(
        f"{ALGORITHM_LABELS[algo]}: ACC {results[algo]['acc']:.2f}%"
        for algo in algorithms
    )
    axis.text(
        0.99,
        0.01,
        metric_text,
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#BBBBBB",
            "alpha": 0.88,
        },
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def export_interactive_centroids_html(
    dataset: str,
    embedding: np.ndarray,
    plot_indices: np.ndarray,
    y_true: np.ndarray,
    results: dict[str, dict],
    class_names: list[str],
    output_path: Path,
) -> None:
    shown_true = y_true[plot_indices].astype(int)
    n_classes = len(np.unique(y_true))
    cmap, color_norm = categorical_colors(n_classes)
    colors = [
        to_hex(cmap(color_norm(class_idx)), keep_alpha=False)
        for class_idx in range(n_classes)
    ]
    sources = [("Ground Truth", None, "star")] + [
        (ALGORITHM_LABELS[algo], algo, marker)
        for algo, marker in zip(
            [algo for algo in ALGORITHM_ORDER if algo in results],
            ["x", "plus", "diamond", "triangle", "square"],
        )
    ]
    centers = []
    for source_name, algorithm, marker in sources:
        center_labels = (
            shown_true
            if algorithm is None
            else results[algorithm]["labels"][plot_indices].astype(int)
        )
        for class_idx in np.unique(center_labels):
            points = embedding[center_labels == class_idx]
            if not len(points):
                continue
            center = points.mean(axis=0)
            centers.append({
                "source": source_name,
                "classId": int(class_idx),
                "x": float(center[0]),
                "y": float(center[1]),
                "marker": marker,
                "acc": (
                    None
                    if algorithm is None
                    else round(float(results[algorithm]["acc"]), 4)
                ),
                "memoryPeakMb": (
                    None
                    if algorithm is None
                    else results[algorithm].get("memory_peak_mb")
                ),
                "memoryDeltaMb": (
                    None
                    if algorithm is None
                    else results[algorithm].get("memory_delta_mb")
                ),
            })

    payload = {
        "dataset": dataset,
        "points": [
            [float(x), float(y), int(label)]
            for (x, y), label in zip(embedding, shown_true)
        ],
        "centers": centers,
        "classNames": [
            class_names[idx] if idx < len(class_names) else f"Class {idx}"
            for idx in range(n_classes)
        ],
        "colors": colors,
        "sources": [source[0] for source in sources],
    }
    data_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    page_title = html.escape(f"{dataset} interactive centroids")
    template = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__PAGE_TITLE__</title>
<style>
  :root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
  * { box-sizing: border-box; }
  body { margin: 0; background: #f5f7f9; color: #18212b; }
  main { min-height: 100vh; display: grid; grid-template-columns: minmax(0, 1fr) 290px; }
  .plot-wrap { min-width: 0; padding: 22px; display: flex; flex-direction: column; }
  h1 { font-size: 20px; margin: 0 0 4px; letter-spacing: 0; }
  .subtitle { color: #65717d; font-size: 13px; margin-bottom: 14px; }
  .canvas-shell { position: relative; flex: 1; min-height: 620px; background: white; border: 1px solid #d9e0e6; }
  canvas { position: absolute; inset: 0; width: 100%; height: 100%; cursor: grab; }
  canvas.dragging { cursor: grabbing; }
  .zoom-controls { position: absolute; top: 12px; left: 12px; z-index: 3; display: flex; gap: 5px; }
  .zoom-controls button { width: 34px; height: 32px; background: #fff; border: 1px solid #cfd7de; cursor: pointer; font-size: 17px; box-shadow: 0 1px 4px #0001; }
  .zoom-controls button:hover { background: #eef2f5; }
  .zoom-controls .reset { width: auto; padding: 0 10px; font-size: 12px; }
  aside { border-left: 1px solid #d9e0e6; background: white; padding: 20px 16px; overflow: auto; }
  h2 { font-size: 13px; text-transform: uppercase; color: #68737e; margin: 4px 0 10px; letter-spacing: .04em; }
  .class-list, .source-list { display: grid; gap: 4px; margin-bottom: 22px; }
  button { border: 0; background: transparent; color: inherit; font: inherit; }
  .class-item { width: 100%; display: flex; align-items: center; gap: 9px; padding: 7px 8px; cursor: pointer; text-align: left; }
  .class-item:hover, .class-item.active { background: #eef2f5; }
  .swatch { width: 12px; height: 12px; flex: 0 0 12px; border-radius: 50%; }
  .all { font-weight: 650; border-bottom: 1px solid #e7ebef; margin-bottom: 5px; }
  .source { display: grid; grid-template-columns: 22px 1fr; gap: 8px; align-items: center; font-size: 13px; padding: 4px 8px; }
  .symbol { text-align: center; font-size: 18px; font-weight: 800; }
  .distance-list { display: grid; gap: 8px; margin-bottom: 22px; }
  .distance-row { display: grid; grid-template-columns: 76px 1fr 48px; gap: 7px; align-items: center; font-size: 12px; }
  .distance-track { height: 8px; background: #edf0f2; overflow: hidden; }
  .distance-bar { height: 100%; min-width: 2px; }
  .distance-value { text-align: right; font-variant-numeric: tabular-nums; }
  .hint { font-size: 12px; color: #68737e; line-height: 1.5; }
  .tooltip { position: fixed; pointer-events: none; display: none; z-index: 4; background: #18212b; color: white; padding: 7px 9px; font-size: 12px; border-radius: 4px; box-shadow: 0 4px 16px #0003; }
  @media (max-width: 850px) {
    main { grid-template-columns: 1fr; }
    aside { border-left: 0; border-top: 1px solid #d9e0e6; }
    .canvas-shell { min-height: 520px; }
  }
</style>
</head>
<body>
<main>
  <section class="plot-wrap">
    <h1 id="title"></h1>
    <div class="subtitle">Ground-truth samples with centroids from all algorithms</div>
    <div class="canvas-shell">
      <canvas id="plot"></canvas>
      <div class="zoom-controls">
        <button id="zoomIn" title="Zoom in">+</button>
        <button id="zoomOut" title="Zoom out">−</button>
        <button id="resetZoom" class="reset" title="Reset view">Reset</button>
      </div>
    </div>
  </section>
  <aside>
    <h2>Classes</h2>
    <div id="classes" class="class-list"></div>
    <h2>Centroid source</h2>
    <div id="sources" class="source-list"></div>
    <h2>Distance ranking</h2>
    <div id="distanceRanking" class="distance-list"></div>
    <div class="hint">Hover a class to isolate it. Click to lock the selection. Use the mouse wheel to zoom, drag to pan, and double-click to reset.</div>
  </aside>
</main>
<div id="tooltip" class="tooltip"></div>
<script>
const DATA = __DATA_JSON__;
const canvas = document.getElementById("plot");
const ctx = canvas.getContext("2d");
const shell = canvas.parentElement;
const tooltip = document.getElementById("tooltip");
const margin = {left: 68, right: 24, top: 22, bottom: 56};
let hoverClass = null, lockedClass = null, hitCenters = [];
let view = {scale:1, offsetX:0, offsetY:0};
let dragging = false, dragStart = null;
const symbols = {star:"★", x:"✕", plus:"✚", diamond:"◆", triangle:"▲", square:"■"};
const sourceColors = {
  "Ground Truth":"#111111", "KMeans":"#D62728", "DisKmeans":"#1F77B4",
  "FDKM":"#2CA02C", "IFDKM":"#9467BD", "IFDFD":"#8C564B"
};
const sourceOffsets = {"KMeans":-28,"DisKmeans":-14,"FDKM":0,"IFDKM":14,"IFDFD":28};
document.getElementById("title").textContent = DATA.dataset + ": interactive centroid comparison";

const xs = DATA.points.map(p => p[0]), ys = DATA.points.map(p => p[1]);
let xmin = Math.min(...xs), xmax = Math.max(...xs), ymin = Math.min(...ys), ymax = Math.max(...ys);
const xpad = (xmax-xmin || 1)*.07, ypad = (ymax-ymin || 1)*.09;
xmin -= xpad; xmax += xpad; ymin -= ypad; ymax += ypad;

function activeClass(){ return lockedClass === null ? hoverClass : lockedClass; }
function baseSx(x){ return margin.left + (x-xmin)/(xmax-xmin)*(canvas.clientWidth-margin.left-margin.right); }
function baseSy(y){ return canvas.clientHeight-margin.bottom-(y-ymin)/(ymax-ymin)*(canvas.clientHeight-margin.top-margin.bottom); }
function plotCenter(){ return {x:(margin.left+canvas.clientWidth-margin.right)/2,y:(margin.top+canvas.clientHeight-margin.bottom)/2}; }
function sx(x){ const c=plotCenter(); return c.x+(baseSx(x)-c.x)*view.scale+view.offsetX; }
function sy(y){ const c=plotCenter(); return c.y+(baseSy(y)-c.y)*view.scale+view.offsetY; }
function resize(){
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(shell.clientWidth*dpr); canvas.height = Math.round(shell.clientHeight*dpr);
  ctx.setTransform(dpr,0,0,dpr,0,0); draw();
}
function tickValues(min,max,count=7){ const out=[]; for(let i=0;i<count;i++) out.push(min+(max-min)*i/(count-1)); return out; }
function drawAxes(){
  const w=canvas.clientWidth,h=canvas.clientHeight;
  ctx.strokeStyle="#cfd6dc"; ctx.lineWidth=1; ctx.fillStyle="#5f6b76"; ctx.font="11px system-ui";
  for(const value of tickValues(xmin,xmax)){
    const x=sx(value); ctx.beginPath();ctx.moveTo(x,margin.top);ctx.lineTo(x,h-margin.bottom);ctx.strokeStyle="#edf0f2";ctx.stroke();
    ctx.fillStyle="#5f6b76";ctx.textAlign="center";ctx.fillText(value.toFixed(2),x,h-margin.bottom+19);
  }
  for(const value of tickValues(ymin,ymax)){
    const y=sy(value);ctx.beginPath();ctx.moveTo(margin.left,y);ctx.lineTo(w-margin.right,y);ctx.strokeStyle="#edf0f2";ctx.stroke();
    ctx.fillStyle="#5f6b76";ctx.textAlign="right";ctx.fillText(value.toFixed(2),margin.left-8,y+4);
  }
  ctx.strokeStyle="#89949e";ctx.beginPath();ctx.moveTo(margin.left,margin.top);ctx.lineTo(margin.left,h-margin.bottom);ctx.lineTo(w-margin.right,h-margin.bottom);ctx.stroke();
  ctx.fillStyle="#36414b";ctx.textAlign="center";ctx.font="12px system-ui";ctx.fillText("PCA component 1",(margin.left+w-margin.right)/2,h-14);
  ctx.save();ctx.translate(18,(margin.top+h-margin.bottom)/2);ctx.rotate(-Math.PI/2);ctx.fillText("PCA component 2",0,0);ctx.restore();
}
function draw(){
  const w=canvas.clientWidth,h=canvas.clientHeight, active=activeClass();
  ctx.clearRect(0,0,w,h);ctx.fillStyle="white";ctx.fillRect(0,0,w,h);drawAxes();
  for(const p of DATA.points){
    if(active!==null && p[2]!==active) continue;
    ctx.globalAlpha=.62;ctx.fillStyle=DATA.colors[p[2]];ctx.beginPath();ctx.arc(sx(p[0]),sy(p[1]),2.3,0,Math.PI*2);ctx.fill();
  }
  ctx.globalAlpha=1; hitCenters=[];
  const groundCenters = new Map(
    DATA.centers
      .filter(c=>c.source==="Ground Truth")
      .map(c=>[c.classId,c])
  );
  for(const c of DATA.centers){
    if(c.source==="Ground Truth" || (active!==null && c.classId!==active)) continue;
    const ground=groundCenters.get(c.classId);
    if(!ground) continue;
    const x1=sx(ground.x),y1=sy(ground.y),x2=sx(c.x),y2=sy(c.y);
    const distance=Math.hypot(c.x-ground.x,c.y-ground.y);
    const weight=1/(1+distance);
    ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);
    ctx.strokeStyle=sourceColors[c.source]||DATA.colors[c.classId];ctx.globalAlpha=.72;ctx.lineWidth=1.8;ctx.stroke();
    if(active!==null || DATA.classNames.length<=10){
      const dx=x2-x1,dy=y2-y1,length=Math.hypot(dx,dy)||1;
      const offset=sourceOffsets[c.source]||0;
      const mx=(x1+x2)/2-dy/length*offset,my=(y1+y2)/2+dx/length*offset;
      const text=`${c.source}: d=${distance.toFixed(2)}  w=${weight.toFixed(2)}`;
      ctx.globalAlpha=.9;ctx.font="10px system-ui";ctx.textAlign="center";ctx.textBaseline="middle";
      const width=ctx.measureText(text).width+8;
      ctx.fillStyle="white";ctx.fillRect(mx-width/2,my-8,width,16);
      ctx.fillStyle=sourceColors[c.source]||"#303a44";ctx.fillText(text,mx,my);
    }
  }
  ctx.globalAlpha=1;
  for(const c of DATA.centers){
    if(active!==null && c.classId!==active) continue;
    const x=sx(c.x),y=sy(c.y);
    ctx.font="bold 22px system-ui";ctx.textAlign="center";ctx.textBaseline="middle";
    ctx.lineWidth=3;ctx.strokeStyle="white";ctx.strokeText(symbols[c.marker],x,y);
    ctx.lineWidth=1.4;ctx.strokeStyle="#111";ctx.strokeText(symbols[c.marker],x,y);
    ctx.fillStyle=DATA.colors[c.classId];ctx.fillText(symbols[c.marker],x,y);
    hitCenters.push({...c,px:x,py:y});
  }
}
function zoomAt(factor, px, py){
  const oldScale=view.scale;
  const newScale=Math.max(.5,Math.min(25,oldScale*factor));
  factor=newScale/oldScale;
  const c=plotCenter();
  view.offsetX=(px-c.x)-((px-c.x)-view.offsetX)*factor;
  view.offsetY=(py-c.y)-((py-c.y)-view.offsetY)*factor;
  view.scale=newScale;draw();
}
function resetView(){ view={scale:1,offsetX:0,offsetY:0};draw(); }
function updateDistanceRanking(){
  const box=document.getElementById("distanceRanking"),selected=activeClass();
  box.innerHTML="";
  if(selected===null){
    box.innerHTML='<div class="hint">Select a class to compare centroid distances.</div>';
    return;
  }
  const ground=DATA.centers.find(c=>c.source==="Ground Truth"&&c.classId===selected);
  if(!ground)return;
  const rows=DATA.centers.filter(c=>c.source!=="Ground Truth"&&c.classId===selected)
    .map(c=>({...c,distance:Math.hypot(c.x-ground.x,c.y-ground.y)}))
    .sort((a,b)=>a.distance-b.distance);
  const max=Math.max(...rows.map(r=>r.distance),.000001);
  rows.forEach(row=>{
    const el=document.createElement("div");el.className="distance-row";
    el.innerHTML=`<span>${row.source}</span><span class="distance-track"><span class="distance-bar" style="display:block;width:${row.distance/max*100}%;background:${sourceColors[row.source]}"></span></span><span class="distance-value">${row.distance.toFixed(3)}</span>`;
    box.appendChild(el);
  });
}
function setActiveUI(){
  document.querySelectorAll(".class-item").forEach(el=>el.classList.toggle("active", Number(el.dataset.id)===activeClass() || (el.dataset.id==="all" && activeClass()===null)));
  updateDistanceRanking();
}
const classBox=document.getElementById("classes");
const allBtn=document.createElement("button");allBtn.className="class-item all active";allBtn.dataset.id="all";allBtn.textContent="All classes";
allBtn.onclick=()=>{lockedClass=null;hoverClass=null;setActiveUI();draw();};classBox.appendChild(allBtn);
DATA.classNames.forEach((name,id)=>{
  const b=document.createElement("button");b.className="class-item";b.dataset.id=id;
  b.innerHTML=`<span class="swatch" style="background:${DATA.colors[id]}"></span><span>${name}</span>`;
  b.onmouseenter=()=>{if(lockedClass===null){hoverClass=id;setActiveUI();draw();}};
  b.onmouseleave=()=>{if(lockedClass===null){hoverClass=null;setActiveUI();draw();}};
  b.onclick=()=>{lockedClass=lockedClass===id?null:id;hoverClass=null;setActiveUI();draw();};classBox.appendChild(b);
});
const sourceBox=document.getElementById("sources");
DATA.sources.forEach((name,i)=>{const d=document.createElement("div");d.className="source";d.innerHTML=`<span class="symbol" style="color:${sourceColors[name]}">${symbols[["star","x","plus","diamond","triangle","square"][i]]}</span><span>${name}</span>`;sourceBox.appendChild(d);});
canvas.onmousemove=e=>{
  if(dragging){
    view.offsetX=dragStart.offsetX+(e.clientX-dragStart.x);
    view.offsetY=dragStart.offsetY+(e.clientY-dragStart.y);
    draw();return;
  }
  const hit=hitCenters.find(c=>Math.hypot(c.px-e.offsetX,c.py-e.offsetY)<12);
  if(!hit){tooltip.style.display="none";return;}
  const acc=hit.acc===null?"":`<br>ACC: ${hit.acc.toFixed(2)}%`;
  const ground=DATA.centers.find(c=>c.source==="Ground Truth" && c.classId===hit.classId);
  const distance=(hit.source==="Ground Truth" || !ground)?null:Math.hypot(hit.x-ground.x,hit.y-ground.y);
  const relation=distance===null?"":`<br>Distance: ${distance.toFixed(4)}<br>Weight: ${(1/(1+distance)).toFixed(4)}`;
  const memory=hit.memoryPeakMb==null?"":`<br>Peak RSS: ${Number(hit.memoryPeakMb).toFixed(2)} MB<br>Added RSS: ${Number(hit.memoryDeltaMb).toFixed(2)} MB`;
  tooltip.innerHTML=`<b>${hit.source}</b><br>${DATA.classNames[hit.classId]}<br>PCA: (${hit.x.toFixed(3)}, ${hit.y.toFixed(3)})${acc}${memory}${relation}`;
  tooltip.style.display="block";tooltip.style.left=(e.clientX+14)+"px";tooltip.style.top=(e.clientY+14)+"px";
};
canvas.onmousedown=e=>{
  if(e.button!==0)return;
  dragging=true;canvas.classList.add("dragging");
  dragStart={x:e.clientX,y:e.clientY,offsetX:view.offsetX,offsetY:view.offsetY};
  tooltip.style.display="none";
};
window.addEventListener("mouseup",()=>{dragging=false;canvas.classList.remove("dragging");});
canvas.onmouseleave=()=>{tooltip.style.display="none";};
canvas.addEventListener("wheel",e=>{
  e.preventDefault();
  const rect=canvas.getBoundingClientRect();
  zoomAt(e.deltaY<0?1.18:1/1.18,e.clientX-rect.left,e.clientY-rect.top);
},{passive:false});
canvas.ondblclick=resetView;
document.getElementById("zoomIn").onclick=()=>{const c=plotCenter();zoomAt(1.25,c.x,c.y);};
document.getElementById("zoomOut").onclick=()=>{const c=plotCenter();zoomAt(1/1.25,c.x,c.y);};
document.getElementById("resetZoom").onclick=resetView;
updateDistanceRanking();
new ResizeObserver(resize).observe(shell); resize();
</script>
</body>
</html>"""
    output_path.write_text(
        template.replace("__PAGE_TITLE__", page_title).replace("__DATA_JSON__", data_json),
        encoding="utf-8",
    )


def export_home_html(output_dir: Path, datasets: list[str]) -> None:
    entries = [
        {"name": dataset, "stem": safe_name(dataset)}
        for dataset in datasets
    ]
    data_json = json.dumps(entries, ensure_ascii=False).replace("</", "<\\/")
    template = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Clustering Results</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; font-family: Inter, ui-sans-serif, system-ui, sans-serif; color: #17212b; background: #eef2f5; }
  main { height: 100vh; display: grid; grid-template-columns: 260px minmax(0,1fr); }
  aside { background: #fff; border-right: 1px solid #d5dde4; padding: 18px 12px; overflow: auto; }
  .brand { font-size: 17px; font-weight: 750; padding: 4px 8px 15px; border-bottom: 1px solid #e5e9ed; margin-bottom: 12px; }
  .eyebrow { color: #71808d; font-size: 11px; text-transform: uppercase; letter-spacing: .06em; margin: 12px 8px 7px; }
  button { font: inherit; color: inherit; }
  .dataset { border: 0; background: transparent; width: 100%; text-align: left; padding: 9px 10px; cursor: pointer; border-left: 3px solid transparent; }
  .dataset:hover { background: #f0f4f7; }
  .dataset.active { background: #e8f1f7; border-left-color: #2878b5; font-weight: 650; }
  .content { min-width: 0; display: grid; grid-template-rows: auto auto minmax(0,1fr); }
  header { background: #fff; padding: 15px 20px 11px; border-bottom: 1px solid #d5dde4; display: flex; align-items: center; justify-content: space-between; gap: 15px; }
  h1 { font-size: 19px; margin: 0; letter-spacing: 0; }
  #openExternal { text-decoration: none; color: #17669d; font-size: 13px; border: 1px solid #b9cbd8; padding: 7px 10px; }
  #openExternal:hover { background: #edf5fa; }
  nav { display: flex; gap: 3px; background: #fff; padding: 8px 16px; border-bottom: 1px solid #d5dde4; overflow-x: auto; }
  .tab { border: 0; background: transparent; white-space: nowrap; padding: 8px 11px; cursor: pointer; color: #5d6974; }
  .tab:hover { background: #f0f3f5; }
  .tab.active { color: #17669d; background: #eaf3f8; font-weight: 650; }
  .viewer { min-height: 0; padding: 12px; }
  iframe { display: block; width: 100%; height: 100%; border: 1px solid #cad4dc; background: white; }
  @media (max-width: 760px) {
    main { grid-template-columns: 1fr; grid-template-rows: auto minmax(0,1fr); }
    aside { border-right: 0; border-bottom: 1px solid #d5dde4; padding: 8px; display: flex; overflow-x: auto; }
    .brand, .eyebrow { display: none; }
    .dataset { width: auto; white-space: nowrap; border-left: 0; border-bottom: 3px solid transparent; }
    .dataset.active { border-bottom-color: #2878b5; }
  }
</style>
</head>
<body>
<main>
  <aside>
    <div class="brand">Clustering Results</div>
    <div class="eyebrow">Datasets</div>
    <div id="datasets"></div>
  </aside>
  <section class="content">
    <header><h1 id="title"></h1><a id="openExternal" target="_blank">Open full page</a></header>
    <nav id="tabs"></nav>
    <div class="viewer"><iframe id="viewer" title="Result viewer"></iframe></div>
  </section>
</main>
<script>
const DATASETS = __DATASETS__;
const VIEWS = [
  ["Interactive","interactive_centroids.html"],
  ["Clusters","clusters.png"],
  ["Ground truth + centers","ground_truth_centers.png"],
  ["Metrics","metrics.png"],
  ["Runtime + iterations","runtime_iterations.png"],
  ["Memory","memory_usage.png"],
  ["Centroid distances","centroid_distances.png"]
];
let selectedDataset = 0, selectedView = 0;
const datasetBox=document.getElementById("datasets"),tabBox=document.getElementById("tabs");
function filePath(){const d=DATASETS[selectedDataset],v=VIEWS[selectedView];return `${d.stem}_${v[1]}`;}
function render(){
  document.querySelectorAll(".dataset").forEach((el,i)=>el.classList.toggle("active",i===selectedDataset));
  document.querySelectorAll(".tab").forEach((el,i)=>el.classList.toggle("active",i===selectedView));
  document.getElementById("title").textContent=DATASETS[selectedDataset].name+" · "+VIEWS[selectedView][0];
  const path=filePath();document.getElementById("viewer").src=path;document.getElementById("openExternal").href=path;
  localStorage.setItem("clusterHomeDataset",selectedDataset);localStorage.setItem("clusterHomeView",selectedView);
}
DATASETS.forEach((dataset,index)=>{
  const button=document.createElement("button");button.className="dataset";button.textContent=dataset.name;
  button.onclick=()=>{selectedDataset=index;render();};datasetBox.appendChild(button);
});
VIEWS.forEach((view,index)=>{
  const button=document.createElement("button");button.className="tab";button.textContent=view[0];
  button.onclick=()=>{selectedView=index;render();};tabBox.appendChild(button);
});
selectedDataset=Math.min(Number(localStorage.getItem("clusterHomeDataset")||0),DATASETS.length-1);
selectedView=Math.min(Number(localStorage.getItem("clusterHomeView")||0),VIEWS.length-1);
render();
</script>
</body>
</html>"""
    (output_dir / "home.html").write_text(
        template.replace("__DATASETS__", data_json),
        encoding="utf-8",
    )


def plot_acc_heatmap(all_results: dict[str, dict], output_path: Path) -> None:
    datasets = list(all_results)
    algorithms = [
        algo for algo in ALGORITHM_ORDER
        if any(algo in all_results[dataset] for dataset in datasets)
    ]
    values = np.full((len(datasets), len(algorithms)), np.nan)
    for row_idx, dataset in enumerate(datasets):
        for col_idx, algorithm in enumerate(algorithms):
            if algorithm in all_results[dataset]:
                values[row_idx, col_idx] = all_results[dataset][algorithm]["acc"]

    fig_width = max(9, len(algorithms) * 1.7)
    fig_height = max(4.5, len(datasets) * 0.75)
    fig, axis = plt.subplots(figsize=(fig_width, fig_height))
    image = axis.imshow(values, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    for row_idx in range(len(datasets)):
        for col_idx in range(len(algorithms)):
            value = values[row_idx, col_idx]
            if not np.isnan(value):
                axis.text(
                    col_idx,
                    row_idx,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black",
                )
    axis.set_xticks(
        np.arange(len(algorithms)),
        [ALGORITHM_LABELS[algo] for algo in algorithms],
    )
    axis.set_yticks(np.arange(len(datasets)), datasets)
    axis.set_title("Best-setup ACC comparison", fontsize=14, fontweight="bold")
    fig.colorbar(image, ax=axis, label="ACC (%)", fraction=0.03, pad=0.03)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_saved_data(
    data_path: Path, output_dir: Path, config: dict | None = None
) -> None:
    if not data_path.exists():
        raise FileNotFoundError(f"Saved visualization data not found: {data_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for item in load_visualization_data(data_path):
        dataset = item["dataset"]
        results = item["results"]
        if config is not None:
            dataset_config = next(
                (
                    dataset_cfg
                    for key, dataset_cfg in config["datasets"].items()
                    if dataset in (key, dataset_cfg.get("name", key))
                ),
                None,
            )
            if dataset_config is not None:
                item["class_names"] = class_names_from_dataset(
                    config["data_dir"], dataset_config, item["y_true"]
                )
        file_stem = safe_name(dataset)
        plot_clusters(
            dataset,
            item["embedding"],
            item["plot_indices"],
            item["y_true"],
            results,
            item["class_names"],
            output_dir / f"{file_stem}_clusters.png",
        )
        plot_metrics(dataset, results, output_dir / f"{file_stem}_metrics.png")
        plot_runtime_iterations(
            dataset,
            results,
            output_dir / f"{file_stem}_runtime_iterations.png",
        )
        plot_memory_usage(
            dataset,
            results,
            output_dir / f"{file_stem}_memory_usage.png",
        )
        plot_centroid_distances(
            dataset,
            item["embedding"],
            item["plot_indices"],
            item["y_true"],
            results,
            item["class_names"],
            output_dir / f"{file_stem}_centroid_distances.png",
        )
        plot_ground_truth_with_predicted_centers(
            dataset,
            item["embedding"],
            item["plot_indices"],
            item["y_true"],
            results,
            item["class_names"],
            output_dir / f"{file_stem}_ground_truth_centers.png",
        )
        export_interactive_centroids_html(
            dataset,
            item["embedding"],
            item["plot_indices"],
            item["y_true"],
            results,
            item["class_names"],
            output_dir / f"{file_stem}_interactive_centroids.html",
        )
        all_results[dataset] = results
        print(f"Rendered {dataset} from {data_path.name}")

    plot_acc_heatmap(all_results, output_dir / "all_datasets_acc.png")
    export_home_html(output_dir, list(all_results))
    print(f"Visualizations -> {output_dir}")


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    if args.render_only:
        data_path = Path(args.render_only).expanduser().resolve()
        config_path = (
            Path(args.config).expanduser().resolve()
            if args.config
            else script_dir / "config.yaml"
        )
        config = None
        if config_path.exists():
            with config_path.open(encoding="utf-8-sig") as f:
                config = yaml.safe_load(f)
        output_dir = (
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir
            else data_path.parent
        )
        render_saved_data(data_path, output_dir, config)
        return

    if not args.best_setups:
        raise ValueError("best_setups.csv is required unless --render-only is used")

    setup_path = Path(args.best_setups).expanduser().resolve()
    config_path = (
        Path(args.config).expanduser().resolve()
        if args.config
        else script_dir / "config.yaml"
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else setup_path.parent / "visualizations"
    )

    if not setup_path.exists():
        raise FileNotFoundError(f"best_setups.csv not found: {setup_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found: {config_path}")

    with config_path.open(encoding="utf-8-sig") as f:
        config = yaml.safe_load(f)

    sys.path.insert(0, str(script_dir))
    from src.algorithms import ALGO_REGISTRY, KERNEL_ALGOS
    from src.data_loaders import load_dataset
    from src.metrics import compute_all
    from src.preprocessing import apply_normalization, make_random_init, stratified_sample

    dataset_configs = {}
    dataset_names_by_key = {}
    for key, dataset_config in config["datasets"].items():
        name = dataset_config.get("name", key)
        dataset_configs[key] = dataset_config
        dataset_configs[name] = dataset_config
        dataset_names_by_key[key] = name

    setup_rows = read_csv(setup_path)
    selected_algorithms = args.algorithms or ALGORITHM_ORDER
    dataset_filter = set(args.datasets or [])
    dataset_filter.update(
        dataset_names_by_key[value]
        for value in list(dataset_filter)
        if value in dataset_names_by_key
    )
    setup_rows = [
        row for row in setup_rows
        if row["algorithm"] in selected_algorithms
        and (not dataset_filter or row["dataset"] in dataset_filter)
    ]
    if not setup_rows:
        raise ValueError("No matching rows found in best_setups.csv")

    grouped_setups = defaultdict(list)
    for row in setup_rows:
        grouped_setups[row["dataset"]].append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_data_path = output_dir / "visualization_data.npz"
    metrics_csv = output_dir / "rerun_metrics.csv"
    if metrics_csv.exists():
        metrics_csv.unlink()

    settings = config["settings"]
    max_iter = int(settings.get("max_iter", 300))
    tol = float(settings.get("tol", 1e-6))
    sample_seed = int(settings.get("seed_base", 42))
    n_init = int(
        args.n_init
        if args.n_init is not None
        else settings.get("n_init_per_run", 1)
    )
    all_results = {}
    saved_datasets = []

    for dataset, rows in grouped_setups.items():
        dataset_config = dataset_configs.get(dataset)
        if dataset_config is None:
            raise KeyError(f"Dataset '{dataset}' is missing from {config_path}")

        print(f"\n[{dataset}] loading data...")
        X_raw, y = load_dataset(config["data_dir"], dataset_config)
        X_raw, y = stratified_sample(
            X_raw,
            y,
            int(dataset_config.get("max_samples", 0)),
            seed=sample_seed,
        )
        n_clusters = int(dataset_config["n_clusters"])
        embedding, plot_indices = shared_pca_embedding(
            X_raw, y, args.max_plot_points, sample_seed
        )
        class_names = class_names_from_dataset(config["data_dir"], dataset_config, y)
        dataset_results = {}

        rows.sort(key=lambda row: ALGORITHM_ORDER.index(row["algorithm"]))
        for row in rows:
            algorithm_name = row["algorithm"]
            norm_mode = row["norm_mode"]
            gamma = parse_gamma(row.get("gamma"))
            seed = int(row["best_seed"])
            X = apply_normalization(X_raw, norm_mode)
            AlgoClass = ALGO_REGISTRY[algorithm_name]

            print(
                f"  {ALGORITHM_LABELS[algorithm_name]:<10} "
                f"mode={norm_mode:<14} gamma={gamma if gamma is not None else '-'} "
                f"seed={seed}"
            )
            if algorithm_name in KERNEL_ALGOS:
                (labels, runtime, n_iter), memory = measure_peak_memory(
                    lambda: run_kernel(
                        AlgoClass,
                        X,
                        n_clusters,
                        gamma,
                        seed,
                        n_init,
                        max_iter,
                        tol,
                        make_random_init,
                    )
                )
            else:
                algorithm = AlgoClass(
                    n_clusters=n_clusters, max_iter=max_iter, tol=tol
                )
                algorithm._seed = seed
                (labels, runtime), memory = measure_peak_memory(
                    lambda: algorithm.fit(X)
                )
                n_iter = int(algorithm.n_iter_)

            metrics = compute_all(y, labels)
            aligned_labels = align_labels(y, labels)
            result = {
                "labels": aligned_labels,
                "norm_mode": norm_mode,
                "gamma": gamma,
                "gamma_text": "-" if gamma is None else f"{gamma:g}",
                "seed": seed,
                "runtime_s": runtime,
                "n_iter": n_iter,
                **memory,
                "acc": metrics["acc"] * 100,
                "nmi": metrics["nmi"] * 100,
                "ari": metrics["ari"] * 100,
            }
            dataset_results[algorithm_name] = result
            append_csv(
                metrics_csv,
                {
                    "dataset": dataset,
                    "algorithm": algorithm_name,
                    "norm_mode": norm_mode,
                    "gamma": "" if gamma is None else gamma,
                    "seed": seed,
                    "acc": f"{result['acc']:.4f}",
                    "nmi": f"{result['nmi']:.4f}",
                    "ari": f"{result['ari']:.4f}",
                    "runtime_s": f"{runtime:.4f}",
                    "n_iter": n_iter,
                    "memory_baseline_mb": f"{memory['memory_baseline_mb']:.4f}",
                    "memory_peak_mb": f"{memory['memory_peak_mb']:.4f}",
                    "memory_delta_mb": f"{memory['memory_delta_mb']:.4f}",
                },
            )

        all_results[dataset] = dataset_results
        file_stem = safe_name(dataset)
        saved_datasets.append({
            "dataset": dataset,
            "embedding": embedding,
            "plot_indices": plot_indices,
            "y_true": y,
            "class_names": class_names,
            "results": dataset_results,
        })
        save_visualization_data(saved_data_path, saved_datasets)
        plot_clusters(
            dataset,
            embedding,
            plot_indices,
            y,
            dataset_results,
            class_names,
            output_dir / f"{file_stem}_clusters.png",
        )
        plot_metrics(
            dataset,
            dataset_results,
            output_dir / f"{file_stem}_metrics.png",
        )
        plot_runtime_iterations(
            dataset,
            dataset_results,
            output_dir / f"{file_stem}_runtime_iterations.png",
        )
        plot_memory_usage(
            dataset,
            dataset_results,
            output_dir / f"{file_stem}_memory_usage.png",
        )
        plot_centroid_distances(
            dataset,
            embedding,
            plot_indices,
            y,
            dataset_results,
            class_names,
            output_dir / f"{file_stem}_centroid_distances.png",
        )
        plot_ground_truth_with_predicted_centers(
            dataset,
            embedding,
            plot_indices,
            y,
            dataset_results,
            class_names,
            output_dir / f"{file_stem}_ground_truth_centers.png",
        )
        export_interactive_centroids_html(
            dataset,
            embedding,
            plot_indices,
            y,
            dataset_results,
            class_names,
            output_dir / f"{file_stem}_interactive_centroids.html",
        )

    plot_acc_heatmap(all_results, output_dir / "all_datasets_acc.png")
    export_home_html(output_dir, list(all_results))
    save_visualization_data(saved_data_path, saved_datasets)
    print(f"\nMetrics -> {metrics_csv}")
    print(f"Saved plot data -> {saved_data_path}")
    print(f"Visualizations -> {output_dir}")


if __name__ == "__main__":
    main()
