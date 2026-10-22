#!/usr/bin/env python3
"""Build a global Rust-vs-C++ timing comparison figure across example outputs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def discover_result_files(root: Path) -> List[Path]:
    out: List[Path] = []
    out.extend(root.glob("examples/option*/output/results.json"))
    out.extend(root.glob("examples/integrations/*/output/results.json"))
    return sorted(set(out))


def load_example_cases(result_files: List[Path]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for p in result_files:
        payload = json.loads(p.read_text(encoding="utf-8"))
        for row in payload.get("results", []):
            item = dict(row)
            item["source_file"] = str(p)
            out.append(item)
    return out


def load_reference_benchmark_cases(root: Path) -> List[Dict[str, object]]:
    ref = root / "reference" / "benchmark_results.json"
    if not ref.exists():
        return []

    payload = json.loads(ref.read_text(encoding="utf-8"))
    out: List[Dict[str, object]] = []
    for case in payload.get("cases", []):
        case_meta = case.get("case", {})
        ops = case.get("operations", {})
        if "train" not in ops or "predict" not in ops:
            continue

        out.append(
            {
                "dataset": str(case_meta.get("dataset", "unknown")),
                "kernel": int(case_meta.get("kernel", 0)),
                "svm_type": int(case_meta.get("svm_type", 0)),
                "c_value": 1.0,
                "train_rows": 0,
                "timing": {
                    "train": ops["train"],
                    "predict": ops["predict"],
                },
                "source_file": str(ref),
            }
        )
    return out


def unique_case_key(row: Dict[str, object]) -> Tuple[str, int, int, float, int]:
    return (
        str(row["dataset"]),
        int(row["kernel"]),
        int(row["svm_type"]),
        float(row["c_value"]),
        int(row["train_rows"]),
    )


def dedupe_cases(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    by_key: Dict[Tuple[str, int, int, float, int], Dict[str, object]] = {}
    for row in rows:
        by_key[unique_case_key(row)] = row
    out = list(by_key.values())
    out.sort(key=lambda r: (str(r["dataset"]), int(r["kernel"]), float(r["c_value"]), int(r["train_rows"])))
    return out


def read_samples(op: Dict[str, object], impl: str) -> List[float]:
    impl_data = op.get(impl, {})
    if not isinstance(impl_data, dict):
        return []
    raw = impl_data.get("samples_ms")
    if isinstance(raw, list) and raw:
        return [float(x) for x in raw]
    med = impl_data.get("median_ms")
    if med is None:
        return []
    return [float(med)]


def median_of(op: Dict[str, object], impl: str) -> float:
    impl_data = op.get(impl, {})
    if not isinstance(impl_data, dict):
        return float("nan")
    med = impl_data.get("median_ms")
    if med is not None:
        return float(med)
    samples = read_samples(op, impl)
    if not samples:
        return float("nan")
    return float(np.median(samples))


def bootstrap_ci_median(values: np.ndarray, rng: np.random.Generator, n_boot: int = 2000) -> Tuple[float, float]:
    if values.size < 2:
        v = float(values[0]) if values.size == 1 else float("nan")
        return v, v
    boots = np.empty(n_boot, dtype=float)
    n = values.size
    for i in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        boots[i] = float(np.median(sample))
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


@dataclass
class CasePoint:
    label: str
    dataset: str
    source: str
    train_rust: float
    train_c: float
    pred_rust: float
    pred_c: float
    train_ratio: float
    pred_ratio: float
    train_run_count: int
    pred_run_count: int


def rows_to_points(rows: List[Dict[str, object]]) -> List[CasePoint]:
    out: List[CasePoint] = []
    for row in rows:
        timing = row.get("timing", {})
        if not isinstance(timing, dict):
            continue
        train = timing.get("train")
        predict = timing.get("predict")
        if not isinstance(train, dict) or not isinstance(predict, dict):
            continue

        train_r = median_of(train, "rust")
        train_c = median_of(train, "c")
        pred_r = median_of(predict, "rust")
        pred_c = median_of(predict, "c")
        if not np.isfinite(train_r) or not np.isfinite(train_c) or not np.isfinite(pred_r) or not np.isfinite(pred_c):
            continue
        if train_c <= 0.0 or pred_c <= 0.0:
            continue

        dataset = str(row.get("dataset", "unknown"))
        kernel = int(row.get("kernel", -1))
        c_value = float(row.get("c_value", 0.0))
        train_rows = int(row.get("train_rows", 0))

        label = f"{dataset}|k{kernel}|C={c_value:g}|n={train_rows}"

        train_samples_r = read_samples(train, "rust")
        train_samples_c = read_samples(train, "c")
        pred_samples_r = read_samples(predict, "rust")
        pred_samples_c = read_samples(predict, "c")

        out.append(
            CasePoint(
                label=label,
                dataset=dataset,
                source=str(row.get("source_file", "unknown")),
                train_rust=train_r,
                train_c=train_c,
                pred_rust=pred_r,
                pred_c=pred_c,
                train_ratio=train_r / train_c,
                pred_ratio=pred_r / pred_c,
                train_run_count=max(1, min(len(train_samples_r), len(train_samples_c))),
                pred_run_count=max(1, min(len(pred_samples_r), len(pred_samples_c))),
            )
        )
    return out


def write_summary(points: List[CasePoint], out_json: Path) -> None:
    rng = np.random.default_rng(20260211)
    train_ratios = np.array([p.train_ratio for p in points], dtype=float)
    pred_ratios = np.array([p.pred_ratio for p in points], dtype=float)

    t_lo, t_hi = bootstrap_ci_median(train_ratios, rng)
    p_lo, p_hi = bootstrap_ci_median(pred_ratios, rng)

    summary = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "case_count": len(points),
        "sources": sorted({p.source for p in points}),
        "runs": {
            "train_min": int(min(p.train_run_count for p in points)),
            "train_max": int(max(p.train_run_count for p in points)),
            "predict_min": int(min(p.pred_run_count for p in points)),
            "predict_max": int(max(p.pred_run_count for p in points)),
        },
        "train_ratio": {
            "median": float(np.median(train_ratios)),
            "p05": float(np.percentile(train_ratios, 5.0)),
            "p95": float(np.percentile(train_ratios, 95.0)),
            "median_ci95_low": t_lo,
            "median_ci95_high": t_hi,
            "rust_faster_fraction": float(np.mean(train_ratios < 1.0)),
        },
        "predict_ratio": {
            "median": float(np.median(pred_ratios)),
            "p05": float(np.percentile(pred_ratios, 5.0)),
            "p95": float(np.percentile(pred_ratios, 95.0)),
            "median_ci95_low": p_lo,
            "median_ci95_high": p_hi,
            "rust_faster_fraction": float(np.mean(pred_ratios < 1.0)),
        },
    }
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def make_plot(points: List[CasePoint], out_png: Path) -> None:
    if not points:
        raise RuntimeError("No timing rows available; run at least one option first.")

    datasets = sorted({p.dataset for p in points})
    cmap = plt.get_cmap("tab10", max(3, len(datasets)))
    color_of = {ds: cmap(i) for i, ds in enumerate(datasets)}

    train_ratios = np.array([p.train_ratio for p in points], dtype=float)
    pred_ratios = np.array([p.pred_ratio for p in points], dtype=float)

    rng = np.random.default_rng(20260211)
    t_lo, t_hi = bootstrap_ci_median(train_ratios, rng)
    p_lo, p_hi = bootstrap_ci_median(pred_ratios, rng)

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.5))

    ax = axes[0, 0]
    for ds in datasets:
        sub = [p for p in points if p.dataset == ds]
        ax.scatter(
            [p.train_c for p in sub],
            [p.train_rust for p in sub],
            s=26,
            alpha=0.8,
            color=color_of[ds],
            label=ds,
        )
    limits = [
        min(min(p.train_c for p in points), min(p.train_rust for p in points)),
        max(max(p.train_c for p in points), max(p.train_rust for p in points)),
    ]
    ax.plot(limits, limits, "k--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("C++ train median (ms)")
    ax.set_ylabel("Rust train median (ms)")
    ax.set_title("Train: C++ vs Rust (per case)")
    ax.grid(True, alpha=0.25)

    ax = axes[0, 1]
    for ds in datasets:
        sub = [p for p in points if p.dataset == ds]
        ax.scatter(
            [p.pred_c for p in sub],
            [p.pred_rust for p in sub],
            s=26,
            alpha=0.8,
            color=color_of[ds],
            label=ds,
        )
    limits = [
        min(min(p.pred_c for p in points), min(p.pred_rust for p in points)),
        max(max(p.pred_c for p in points), max(p.pred_rust for p in points)),
    ]
    ax.plot(limits, limits, "k--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("C++ predict median (ms)")
    ax.set_ylabel("Rust predict median (ms)")
    ax.set_title("Predict: C++ vs Rust (per case)")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    sort_idx = np.argsort(train_ratios)
    ax.plot(train_ratios[sort_idx], label="Train Rust/C++", linewidth=1.4)
    sort_idx_p = np.argsort(pred_ratios)
    ax.plot(pred_ratios[sort_idx_p], label="Predict Rust/C++", linewidth=1.4)
    ax.axhline(1.0, color="k", linestyle="--", linewidth=1)
    ax.set_ylabel("Ratio")
    ax.set_xlabel("Sorted case index")
    ax.set_title("Per-case Rust/C++ Ratios")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax = axes[1, 1]
    bins = max(12, min(28, int(np.sqrt(len(points)) * 2)))
    ax.hist(train_ratios, bins=bins, alpha=0.58, label="Train", color="#1f77b4")
    ax.hist(pred_ratios, bins=bins, alpha=0.58, label="Predict", color="#ff7f0e")
    ax.axvline(1.0, color="k", linestyle="--", linewidth=1)
    ax.axvline(float(np.median(train_ratios)), color="#1f77b4", linestyle=":", linewidth=1.3)
    ax.axvline(float(np.median(pred_ratios)), color="#ff7f0e", linestyle=":", linewidth=1.3)
    ax.set_xlabel("Rust/C++ ratio")
    ax.set_ylabel("Case count")
    ax.set_title("Ratio Distribution")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=color_of[ds], label=ds)
        for ds in datasets[:8]
    ]
    if legend_handles:
        axes[0, 0].legend(handles=legend_handles, loc="upper left", fontsize=8, framealpha=0.9)

    run_min = min(min(p.train_run_count, p.pred_run_count) for p in points)
    run_max = max(max(p.train_run_count, p.pred_run_count) for p in points)

    fig.suptitle(
        (
            f"Rust vs C++ Timing Comparison ({len(points)} cases, runs/case={run_min}-{run_max})\n"
            f"Median train ratio={np.median(train_ratios):.3f} (95% CI [{t_lo:.3f}, {t_hi:.3f}]), "
            f"median predict ratio={np.median(pred_ratios):.3f} (95% CI [{p_lo:.3f}, {p_hi:.3f}])"
        ),
        fontsize=12,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=320, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repository root")
    parser.add_argument("--out", default="examples/comparison.png", help="Output figure path")
    parser.add_argument(
        "--summary",
        default="examples/comparison_summary.json",
        help="Output summary JSON path",
    )
    parser.add_argument(
        "--min-runs",
        type=int,
        default=3,
        help="Minimum sample count per operation to include a case in figure/summary",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_png = (root / args.out).resolve()
    out_json = (root / args.summary).resolve()

    result_files = discover_result_files(root)
    rows: List[Dict[str, object]] = []
    if result_files:
        rows.extend(load_example_cases(result_files))
    rows.extend(load_reference_benchmark_cases(root))
    if not rows:
        raise RuntimeError(
            "No timing data found. Run an example option or ensure reference/benchmark_results.json exists."
        )

    points = rows_to_points(dedupe_cases(rows))
    points = [p for p in points if min(p.train_run_count, p.pred_run_count) >= args.min_runs]
    if not points:
        raise RuntimeError("No valid timing points extracted from inputs after run-count filtering.")

    make_plot(points, out_png)
    write_summary(points, out_json)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
