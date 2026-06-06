"""Đọc summary.csv đã lưu và in bảng so sánh với paper."""
import sys, csv
from pathlib import Path
import pandas as pd
import numpy as np

# ── Paper targets ────────────────────────────────────────────────────────────
PAPER = {
    "glass+identification": {"kmeans":51.7,"diskmeans":51.2,"fdkm":54.0,"ifdkm":54.2,"ifdfd":53.5},
    "ecoli":                {"kmeans":54.4,"diskmeans":67.0,"fdkm":69.0,"ifdkm":70.8,"ifdfd":72.5},
    "Breast_Cancer":        {"kmeans":85.4,"diskmeans":89.3,"fdkm":93.0,"ifdkm":91.4,"ifdfd":90.7},
}
DATASETS   = list(PAPER.keys())
ALGORITHMS = ["kmeans","diskmeans","fdkm","ifdkm","ifdfd"]

# ── Load summary.csv ──────────────────────────────────────────────────────────
results_dir = sorted(Path("results").iterdir())[-1]
summary_csv = results_dir / "summary.csv"
print(f"Reading: {summary_csv}\n")

df = pd.read_csv(summary_csv)
df["acc"] = pd.to_numeric(df["acc_mean"], errors="coerce")

# Index: (dataset, norm_mode, algorithm) → acc
idx = {}
for _, row in df.iterrows():
    idx[(row["dataset"], row["norm_mode"], row["algorithm"])] = row["acc"]

norm_modes = list(df["norm_mode"].unique())

# ── Compute MAE per mode ──────────────────────────────────────────────────────
mode_stats = {}
for mode in norm_modes:
    diffs = []
    per_ds = {}
    for ds in DATASETS:
        ds_diffs = []
        for algo in ALGORITHMS:
            our   = idx.get((ds, str(mode), algo))
            paper = PAPER.get(ds, {}).get(algo)
            if our is not None and paper is not None:
                diffs.append(abs(our - paper))
                ds_diffs.append(abs(our - paper))
        per_ds[ds] = np.mean(ds_diffs) if ds_diffs else float("nan")
    mode_stats[str(mode)] = {
        "mae": np.mean(diffs) if diffs else float("nan"),
        "per_ds": per_ds,
    }

ranked = sorted(mode_stats.items(), key=lambda x: x[1]["mae"])

# ── Print ranking ─────────────────────────────────────────────────────────────
print("=" * 80)
print("  RANKING: centering modes by total MAE vs paper ACC (%)")
print("  Lower MAE = closer to paper")
print("=" * 80)
print(f"  {'Rank':<5} {'Mode':<16} {'MAE total':>10}  "
      f"{'MAE[glass]':>11}  {'MAE[ecoli]':>11}  {'MAE[BC]':>9}")
print("-" * 80)
for rank, (mode, st) in enumerate(ranked, 1):
    marker = "  <-- BEST" if rank == 1 else ("  <-- 2nd" if rank == 2 else "")
    g = st["per_ds"].get("glass+identification", float("nan"))
    e = st["per_ds"].get("ecoli", float("nan"))
    b = st["per_ds"].get("Breast_Cancer", float("nan"))
    print(f"  {rank:<5} {mode:<16} {st['mae']:>10.2f}  {g:>11.2f}  {e:>11.2f}  {b:>9.2f}{marker}")

print()

# ── Detail for top 3 modes ────────────────────────────────────────────────────
print("=" * 80)
print("  DETAIL — Top 3 modes vs paper")
print("=" * 80)

for rank, (mode, _) in enumerate(ranked[:3], 1):
    print(f"\n  #{rank}: [{mode}]")
    print(f"  {'Dataset':<24} {'Algo':<12} {'Ours':>8} {'Paper':>8} {'Diff':>7}")
    print("  " + "-" * 60)
    for ds in DATASETS:
        for algo in ALGORITHMS:
            our   = idx.get((ds, str(mode), algo))
            paper = PAPER.get(ds, {}).get(algo)
            if our is not None and paper is not None:
                diff = our - paper
                sign = "+" if diff >= 0 else ""
                flag = " ✓" if abs(diff) < 2 else (" ~" if abs(diff) < 5 else "")
                print(f"  {ds:<24} {algo:<12} {our:>8.2f} {paper:>8.1f} {sign}{diff:>6.2f}{flag}")

print()

# ── BC KMeans special check ───────────────────────────────────────────────────
print("=" * 80)
print("  BREAST CANCER — KMeans by mode (paper target: 85.4)")
print("=" * 80)
print(f"  {'Mode':<16} {'KMeans ACC':>12} {'Diff from 85.4':>16}")
print("-" * 50)
bc_km = [(m, idx.get(("Breast_Cancer", str(m), "kmeans"), float("nan")))
         for m in norm_modes]
bc_km.sort(key=lambda x: abs(x[1]-85.4) if not np.isnan(x[1]) else 99)
for m, acc in bc_km:
    diff = acc - 85.4 if not np.isnan(acc) else float("nan")
    flag = " ✓" if abs(diff) < 0.1 else ""
    print(f"  {str(m):<16} {acc:>12.2f} {diff:>+15.2f}{flag}")

# ── Save comparison CSV ───────────────────────────────────────────────────────
out_path = results_dir / "comparison_to_paper.csv"
with open(out_path, "w", newline="", encoding="utf-8") as f:
    fields = ["rank","norm_mode","total_mae"] + [
        f"mae_{d.split('+')[0].split('_')[0]}" for d in DATASETS
    ]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for rank, (mode, st) in enumerate(ranked, 1):
        row = {"rank": rank, "norm_mode": mode,
               "total_mae": f"{st['mae']:.4f}"}
        for ds, key in zip(DATASETS, ["mae_glass","mae_ecoli","mae_Breast"]):
            row[key] = f"{st['per_ds'].get(ds, float('nan')):.4f}"
        w.writerow(row)

print(f"\n  Comparison CSV -> {out_path}")
