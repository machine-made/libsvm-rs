#!/usr/bin/env python3
"""Combine in-process WASM and C++ benchmark outputs into comparison artifacts."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = (len(sorted_vals) - 1) * (p / 100.0)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return sorted_vals[lo]
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def summary(samples_ms: List[float]) -> Dict[str, float | List[float]]:
    ordered = sorted(samples_ms)
    return {
        "samples_ms": [round(v, 6) for v in samples_ms],
        "median_ms": round(statistics.median(samples_ms), 6),
        "p95_ms": round(percentile(ordered, 95.0), 6),
        "mean_ms": round(statistics.mean(samples_ms), 6),
        "min_ms": round(min(samples_ms), 6),
        "max_ms": round(max(samples_ms), 6),
    }


def read_labels(libsvm_file: Path) -> List[float]:
    out: List[float] = []
    with libsvm_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(float(line.split()[0]))
    return out


def classification_metrics(labels: List[float], wasm_preds: List[float], cpp_preds: List[float]) -> Dict[str, float]:
    if not labels or len(labels) != len(wasm_preds) or len(labels) != len(cpp_preds):
        raise RuntimeError("prediction/label size mismatch")
    n = len(labels)
    wasm_ok = sum(1 for y, p in zip(labels, wasm_preds) if abs(y - p) < 1e-12)
    cpp_ok = sum(1 for y, p in zip(labels, cpp_preds) if abs(y - p) < 1e-12)
    agree = sum(1 for pw, pc in zip(wasm_preds, cpp_preds) if abs(pw - pc) < 1e-12)
    return {
        "rust_accuracy": wasm_ok / n,
        "c_accuracy": cpp_ok / n,
        "prediction_agreement": agree / n,
    }


def write_report(
    out_md: Path,
    warmup: int,
    runs: int,
    train_rows: int,
    test_rows: int,
    train_ratio: float,
    pred_ratio: float,
    metrics: Dict[str, float],
) -> None:
    lines = [
        "# WASM Inference Benchmark Results",
        "",
        f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}",
        "",
        "## Methodology",
        "",
        "- Both runtimes use in-process timing for compute-only sections.",
        "- Data parsing/loading is outside timing windows for both runtimes.",
        "- Runtime A (`rust` field in JSON): Rust compiled to `wasm32-unknown-unknown`, executed in Node.js via `wasm-bindgen`.",
        "- Runtime B (`c` field in JSON): C++ LIBSVM via an in-process benchmark harness linked to `vendor/libsvm/svm.cpp`.",
        "",
        "## Setup",
        "",
        "- Dataset: `heart_scale` split (classification)",
        f"- Train rows: {train_rows}",
        f"- Test rows: {test_rows}",
        "- Parameter: C-SVC, RBF kernel, C=1, gamma=1/13",
        f"- Warmup: {warmup}",
        f"- Runs: {runs}",
        "",
        "## Timing Ratios (median)",
        "",
        f"- Train wasm/C++: {train_ratio:.3f}",
        f"- Predict wasm/C++: {pred_ratio:.3f}",
        "",
        "## Correctness",
        "",
        f"- WASM accuracy: {metrics['rust_accuracy']:.4f}",
        f"- C++ accuracy: {metrics['c_accuracy']:.4f}",
        f"- Prediction agreement: {metrics['prediction_agreement']:.4f}",
        "",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")


def make_plot(out_png: Path, wasm_train: List[float], cpp_train: List[float], wasm_pred: List[float], cpp_pred: List[float]) -> None:
    def _box(ax, values: List[List[float]], labels: List[str]) -> None:
        try:
            ax.boxplot(values, tick_labels=labels, showfliers=True)
        except TypeError:
            ax.boxplot(values, labels=labels, showfliers=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))

    ax = axes[0]
    _box(ax, [wasm_train, cpp_train], ["WASM(Node)", "C++"])
    ax.set_title("Train compute time")
    ax.set_ylabel("ms")
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    _box(ax, [wasm_pred, cpp_pred], ["WASM(Node)", "C++"])
    ax.set_title("Predict compute time")
    ax.set_ylabel("ms")
    ax.grid(True, alpha=0.25)

    fig.suptitle("WASM vs C++ in-process timing on heart_scale split")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", required=True)
    parser.add_argument("--wasm-json", required=True)
    parser.add_argument("--cpp-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--runs", type=int, required=True)
    args = parser.parse_args()

    test_file = Path(args.test).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    wasm_payload = json.loads(Path(args.wasm_json).read_text(encoding="utf-8"))
    cpp_payload = json.loads(Path(args.cpp_json).read_text(encoding="utf-8"))

    wasm_train_samples = [float(x) for x in wasm_payload["train_samples_ms"]]
    wasm_pred_samples = [float(x) for x in wasm_payload["predict_samples_ms"]]
    wasm_preds = [float(x) for x in wasm_payload["predictions"]]

    cpp_train_samples = [float(x) for x in cpp_payload["train_samples_ms"]]
    cpp_pred_samples = [float(x) for x in cpp_payload["predict_samples_ms"]]
    cpp_preds = [float(x) for x in cpp_payload["predictions"]]

    labels = read_labels(test_file)
    metrics = classification_metrics(labels, wasm_preds, cpp_preds)

    train_med_wasm = float(statistics.median(wasm_train_samples))
    train_med_cpp = float(statistics.median(cpp_train_samples))
    pred_med_wasm = float(statistics.median(wasm_pred_samples))
    pred_med_cpp = float(statistics.median(cpp_pred_samples))
    gamma = 1.0 / 13.0

    row = {
        "dataset": "heart_scale_wasm",
        "task": "classification",
        "train_rows": int(wasm_payload["train_rows"]),
        "test_rows": int(wasm_payload["test_rows"]),
        "svm_type": 0,
        "kernel": 2,
        "kernel_name": "rbf",
        "c_value": 1.0,
        "gamma": gamma,
        "runtime": "wasm-nodejs-inprocess",
        "timing": {
            "train": {
                "rust": summary(wasm_train_samples),
                "c": summary(cpp_train_samples),
                "rust_over_c_median_ratio": round(train_med_wasm / train_med_cpp, 6),
            },
            "predict": {
                "rust": summary(wasm_pred_samples),
                "c": summary(cpp_pred_samples),
                "rust_over_c_median_ratio": round(pred_med_wasm / pred_med_cpp, 6),
            },
        },
        "metrics": metrics,
        "notes": {
            "rust_impl": "Rust -> WASM via wasm-bindgen, timed inside WASM function",
            "c_impl": "C++ LIBSVM in-process harness linked to svm.cpp",
        },
    }

    out_results = out_dir / "results.json"
    out_results.write_text(
        json.dumps(
            {
                "metadata": {
                    "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "warmup": args.warmup,
                    "runs": args.runs,
                    "test_file": str(test_file),
                },
                "results": [row],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    out_report = out_dir / "report.md"
    write_report(
        out_report,
        warmup=args.warmup,
        runs=args.runs,
        train_rows=int(wasm_payload["train_rows"]),
        test_rows=int(wasm_payload["test_rows"]),
        train_ratio=float(row["timing"]["train"]["rust_over_c_median_ratio"]),
        pred_ratio=float(row["timing"]["predict"]["rust_over_c_median_ratio"]),
        metrics=metrics,
    )

    out_plot = out_dir / "wasm_vs_cpp.png"
    make_plot(out_plot, wasm_train_samples, cpp_train_samples, wasm_pred_samples, cpp_pred_samples)
    print(f"Wrote {out_results}")
    print(f"Wrote {out_report}")
    print(f"Wrote {out_plot}")


if __name__ == "__main__":
    main()
