"""
V4 Runner — Centering Study.

Loops over every (dataset × normalize_mode × algorithm) combination,
compares to paper targets, and writes a ranked comparison table.
"""

from __future__ import annotations

import csv
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from src.algorithms import ALGO_REGISTRY, KERNEL_ALGOS
from src.data_loaders import load_dataset
from src.metrics import compute_all
from src.preprocessing import (
    ALL_MODES,
    apply_normalization,
    make_random_init,
    stratified_sample,
)

logger = logging.getLogger(__name__)


# ── Logging ──────────────────────────────────────────────────────────────────

def setup_logging(log_dir: str, timestamp: str) -> None:
    log_path = Path(log_dir) / f"run_{timestamp}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"))
    root.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"))
    root.addHandler(ch)
    logger.info(f"Log -> {log_path}")


# ── CSV helpers ───────────────────────────────────────────────────────────────

RAW_FIELDS = [
    "dataset", "norm_mode", "algorithm", "run", "seed",
    "gamma", "acc", "nmi", "ari", "runtime_s",
]

SUMMARY_FIELDS = [
    "dataset", "norm_mode", "algorithm",
    "acc_mean", "acc_std", "nmi_mean", "nmi_std",
    "ari_mean", "ari_std", "gamma",
]


def _append_csv(path: Path, row: dict, fields: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_hdr = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_hdr:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})


# ── Main Runner ───────────────────────────────────────────────────────────────

class Runner:

    def __init__(self, config: dict, config_dir: str):
        self.config     = config
        self.config_dir = Path(config_dir)
        s = config["settings"]

        self.n_runs            = int(s["n_runs"])
        self.n_init_per_run    = int(s.get("n_init_per_run", 1))
        self.gamma_grid        = [10.0 ** p for p in s["gamma_grid"]]
        self.gamma_search_runs = int(s.get("gamma_search_runs", 2))
        self.gamma_select_by   = s.get("gamma_select_by", "acc")
        self.max_iter          = int(s.get("max_iter", 300))
        self.tol               = float(s.get("tol", 1e-6))
        self.seed_base         = int(s.get("seed_base", 42))
        self.data_dir          = config["data_dir"]
        self.paper_targets     = config.get("paper_targets", {})

        self.timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root        = self.config_dir / s["output_dir"] / self.timestamp
        out_root.mkdir(parents=True, exist_ok=True)
        self.output_dir = out_root
        self.raw_csv    = out_root / "raw_runs.csv"
        self.raw_run_log_csv = out_root / "raw_run_log.csv"
        self.summary_csv = out_root / "summary.csv"

        setup_logging(str(self.config_dir / s["log_dir"]), self.timestamp)

    # ── Entry point ───────────────────────────────────────────────────────────

    def run_all(self) -> list[dict]:
        """Run all (dataset × norm_mode × algorithm) combinations."""
        all_summaries: list[dict] = []

        for ds_key, ds_cfg in self.config["datasets"].items():
            if not ds_cfg.get("enabled", True):
                continue

            ds_name = ds_cfg.get("name", ds_key)
            norm_modes = ds_cfg.get("normalize_modes",
                                    [ds_cfg.get("normalize", "center")])

            # Load raw data once per dataset
            X_raw, y = load_dataset(self.data_dir, ds_cfg)
            max_s    = int(ds_cfg.get("max_samples", 0))
            X_raw, y = stratified_sample(X_raw, y, max_s, seed=self.seed_base)
            n_clusters = int(ds_cfg["n_clusters"])

            logger.info("")
            logger.info("=" * 72)
            logger.info(f"  DATASET: {ds_name}  (n={X_raw.shape[0]}, d={X_raw.shape[1]}, c={n_clusters})")
            logger.info(f"  Modes to run: {norm_modes}")
            logger.info("=" * 72)

            for mode in norm_modes:
                logger.info(f"\n  ── normalize: [{mode}] ──")
                # Apply normalization; skip if result has NaN/Inf
                X = apply_normalization(X_raw, mode)
                if not np.all(np.isfinite(X)):
                    logger.warning(f"    [{mode}] produced non-finite values — SKIP")
                    continue

                for algo_name in self.config["algorithms"]:
                    is_kernel = algo_name in KERNEL_ALGOS
                    AlgoClass = ALGO_REGISTRY[algo_name]

                    # Gamma search
                    best_gamma = None
                    if is_kernel:
                        best_gamma = self._gamma_search(
                            AlgoClass, X, y, n_clusters, algo_name, ds_name, mode
                        )

                    # n_runs runs
                    accs, nmis, aris = [], [], []
                    for run_idx in range(self.n_runs):
                        seed = self.seed_base + run_idx
                        if is_kernel:
                            labels, t = self._run_kernel(
                                AlgoClass, X, y, n_clusters, best_gamma, seed
                            )
                        else:
                            algo = AlgoClass(n_clusters=n_clusters,
                                             max_iter=self.max_iter, tol=self.tol)
                            algo._seed = seed
                            labels, t = algo.fit(X)

                        m = compute_all(y, labels)
                        accs.append(m["acc"] * 100)
                        nmis.append(m["nmi"] * 100)
                        aris.append(m["ari"] * 100)

                        raw_row = {
                            "dataset": ds_name, "norm_mode": mode,
                            "algorithm": algo_name, "run": run_idx + 1,
                            "seed": seed,
                            "gamma": best_gamma if best_gamma is not None else "",
                            "acc": f"{m['acc']*100:.4f}",
                            "nmi": f"{m['nmi']*100:.4f}",
                            "ari": f"{m['ari']*100:.4f}",
                            "runtime_s": f"{t:.4f}",
                        }
                        _append_csv(self.raw_csv, raw_row, RAW_FIELDS)
                        _append_csv(self.raw_run_log_csv, raw_row, RAW_FIELDS)

                    acc_m, acc_s = float(np.mean(accs)), float(np.std(accs))
                    nmi_m, nmi_s = float(np.mean(nmis)), float(np.std(nmis))
                    ari_m, ari_s = float(np.mean(aris)), float(np.std(aris))

                    logger.info(
                        f"    [{mode}] {algo_name.upper():<10} "
                        f"ACC={acc_m:.2f}±{acc_s:.2f}  "
                        f"NMI={nmi_m:.2f}±{nmi_s:.2f}  "
                        f"ARI={ari_m:.2f}±{ari_s:.2f}"
                    )

                    row = {
                        "dataset": ds_name, "norm_mode": mode,
                        "algorithm": algo_name,
                        "acc_mean": f"{acc_m:.4f}", "acc_std": f"{acc_s:.4f}",
                        "nmi_mean": f"{nmi_m:.4f}", "nmi_std": f"{nmi_s:.4f}",
                        "ari_mean": f"{ari_m:.4f}", "ari_std": f"{ari_s:.4f}",
                        "gamma": best_gamma if best_gamma is not None else "N/A",
                    }
                    _append_csv(self.summary_csv, row, SUMMARY_FIELDS)
                    all_summaries.append({
                        **row,
                        "acc_mean_f": acc_m, "nmi_mean_f": nmi_m, "ari_mean_f": ari_m,
                    })

        return all_summaries

    # ── Gamma search ──────────────────────────────────────────────────────────

    def _gamma_search(self, AlgoClass, X, y, n_clusters,
                      algo_name, ds_name, mode) -> float:
        best_gamma, best_score = self.gamma_grid[0], -1.0
        for gamma in self.gamma_grid:
            scores = []
            for trial in range(self.gamma_search_runs):
                seed   = self.seed_base + trial
                init   = make_random_init(len(y), n_clusters, seed)
                algo   = AlgoClass(n_clusters=n_clusters,
                                   max_iter=self.max_iter, tol=self.tol)
                lbs, _ = algo.fit(X, gamma=gamma, init_labels=init)
                scores.append(compute_all(y, lbs)[self.gamma_select_by])
            mean_s = float(np.mean(scores))
            if mean_s > best_score:
                best_score, best_gamma = mean_s, gamma
        return best_gamma

    # ── Single kernel algo run ────────────────────────────────────────────────

    def _run_kernel(self, AlgoClass, X, y, n_clusters, gamma, outer_seed):
        best_labels, best_obj, best_t = None, -np.inf, 0.0
        for i in range(self.n_init_per_run):
            init_seed = outer_seed + i * 1000
            init      = make_random_init(len(y), n_clusters, init_seed)
            algo      = AlgoClass(n_clusters=n_clusters,
                                  max_iter=self.max_iter, tol=self.tol)
            lbs, t    = algo.fit(X, gamma=gamma, init_labels=init)
            obj       = algo.objective_ if algo.objective_ is not None else -np.inf
            if obj > best_obj:
                best_obj, best_labels, best_t = obj, lbs, t
        return best_labels, best_t


# ── Comparison table ──────────────────────────────────────────────────────────

def build_comparison(summaries: list[dict], paper_targets: dict,
                     output_dir: Path, n_runs: int) -> None:
    """
    For each (dataset, norm_mode): compute total distance to paper's ACC numbers.
    Write ranked comparison table and print the best match.
    """
    if not paper_targets:
        return

    datasets   = list(paper_targets.keys())
    algorithms = ["kmeans", "diskmeans", "fdkm", "ifdkm", "ifdfd"]

    # Index summaries by (dataset, norm_mode, algorithm)
    idx = {}
    for s in summaries:
        idx[(s["dataset"], s["norm_mode"], s["algorithm"])] = s["acc_mean_f"]

    # Available norm modes
    norm_modes = list(dict.fromkeys(s["norm_mode"] for s in summaries))

    # For each mode, compute MAE vs paper (across all datasets × algorithms)
    mode_scores: dict[str, dict] = {}
    for mode in norm_modes:
        total_mae = 0.0
        count     = 0
        per_ds    = {}
        for ds in datasets:
            ds_mae = 0.0
            ds_cnt = 0
            for algo in algorithms:
                paper_val = paper_targets.get(ds, {}).get(algo)
                our_val   = idx.get((ds, mode, algo))
                if paper_val is not None and our_val is not None:
                    diff = abs(our_val - paper_val)
                    total_mae += diff
                    ds_mae    += diff
                    count     += 1
                    ds_cnt    += 1
            per_ds[ds] = ds_mae / ds_cnt if ds_cnt > 0 else float("nan")
        mode_scores[mode] = {
            "total_mae": total_mae / count if count > 0 else float("nan"),
            "per_ds":    per_ds,
        }

    # Sort by total MAE
    ranked = sorted(mode_scores.items(), key=lambda x: x[1]["total_mae"])

    # ── Save comparison CSV ──────────────────────────────────────────────────
    comp_path = output_dir / "comparison_to_paper.csv"
    with open(comp_path, "w", newline="", encoding="utf-8") as f:
        fields = ["rank", "norm_mode", "total_mae_vs_paper"] + [
            f"mae_{ds.replace('+', '_')}" for ds in datasets
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rank, (mode, scores) in enumerate(ranked, 1):
            row = {
                "rank":                 rank,
                "norm_mode":            mode,
                "total_mae_vs_paper":   f"{scores['total_mae']:.4f}",
            }
            for ds in datasets:
                row[f"mae_{ds.replace('+','_')}"] = f"{scores['per_ds'].get(ds, float('nan')):.4f}"
            w.writerow(row)

    # ── Save per-mode ACC table ───────────────────────────────────────────────
    acc_path = output_dir / "acc_by_mode.csv"
    with open(acc_path, "w", newline="", encoding="utf-8") as f:
        # Columns: norm_mode, then for each dataset×algo: our_acc | paper_acc | diff
        header_fields = ["norm_mode"]
        for ds in datasets:
            for algo in algorithms:
                header_fields += [f"{ds}_{algo}_ours", f"{ds}_{algo}_paper",
                                  f"{ds}_{algo}_diff"]
        w = csv.DictWriter(f, fieldnames=header_fields)
        w.writeheader()
        for mode in norm_modes:
            row = {"norm_mode": mode}
            for ds in datasets:
                for algo in algorithms:
                    our   = idx.get((ds, mode, algo))
                    paper = paper_targets.get(ds, {}).get(algo)
                    row[f"{ds}_{algo}_ours"]  = f"{our:.2f}"  if our   is not None else "?"
                    row[f"{ds}_{algo}_paper"] = f"{paper:.1f}" if paper is not None else "?"
                    row[f"{ds}_{algo}_diff"]  = (
                        f"{abs(our - paper):.2f}" if (our is not None and paper is not None) else "?"
                    )
            w.writerow(row)

    # ── Print summary to console ─────────────────────────────────────────────
    print()
    print("=" * 80)
    print(f"  CENTERING STUDY — Ranked by total MAE vs paper ({n_runs} runs)")
    print("=" * 80)
    print(f"  {'Rank':<5} {'Mode':<16} {'Total MAE':>10}", end="")
    for ds in datasets:
        label = ds[:8]
        print(f"  {'MAE['+label+']':>14}", end="")
    print()
    print("-" * 80)
    for rank, (mode, scores) in enumerate(ranked, 1):
        marker = " ◄ BEST" if rank == 1 else ""
        print(f"  {rank:<5} {mode:<16} {scores['total_mae']:>10.2f}", end="")
        for ds in datasets:
            print(f"  {scores['per_ds'].get(ds, float('nan')):>14.2f}", end="")
        print(marker)
    print()

    # ── Print best mode detail ────────────────────────────────────────────────
    best_mode = ranked[0][0]
    print(f"  BEST MODE: [{best_mode}]")
    print()
    print(f"  {'Dataset':<24} {'Algo':<12} {'Ours':>8} {'Paper':>8} {'Diff':>6}")
    print("  " + "-" * 62)
    for ds in datasets:
        for algo in algorithms:
            our   = idx.get((ds, best_mode, algo))
            paper = paper_targets.get(ds, {}).get(algo)
            if our is not None and paper is not None:
                diff  = our - paper
                sign  = "+" if diff >= 0 else ""
                flag  = " ✓" if abs(diff) < 2 else ""
                print(f"  {ds:<24} {algo:<12} {our:>8.2f} {paper:>8.1f} {sign}{diff:>5.2f}{flag}")
    print()
    print(f"  Comparison saved -> {comp_path}")
    print(f"  Full ACC table  -> {acc_path}")
