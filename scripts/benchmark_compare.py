#!/usr/bin/env python3
"""
Benchmark Rust CLI binaries against vendored LIBSVM C binaries.

Outputs:
- reference/benchmark_results.json
- reference/benchmark_report.md
"""

from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class Case:
    svm_type: int
    kernel: int
    dataset: str

    @property
    def id(self) -> str:
        dataset_id = self.dataset.replace(".", "_")
        return f"s{self.svm_type}_t{self.kernel}_{dataset_id}"


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


def run(cmd: List[str], *, quiet: bool = True) -> None:
    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL if quiet else None,
        stderr=subprocess.DEVNULL if quiet else None,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(cmd)}")


def measure_command(cmd: List[str], warmup: int, runs: int) -> List[float]:
    for _ in range(warmup):
        run(cmd, quiet=True)

    samples_ms: List[float] = []
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        run(cmd, quiet=True)
        t1 = time.perf_counter_ns()
        samples_ms.append((t1 - t0) / 1e6)
    return samples_ms


def summary(samples_ms: List[float]) -> Dict[str, float]:
    ordered = sorted(samples_ms)
    return {
        "samples_ms": [round(v, 6) for v in samples_ms],
        "median_ms": round(statistics.median(samples_ms), 6),
        "p95_ms": round(percentile(ordered, 95.0), 6),
        "mean_ms": round(statistics.mean(samples_ms), 6),
        "min_ms": round(min(samples_ms), 6),
        "max_ms": round(max(samples_ms), 6),
    }


def dataset_matrix() -> List[Case]:
    cases: List[Case] = []

    # Classification datasets for C-SVC / nu-SVC / one-class.
    class_sets = ["heart_scale", "iris.scale"]
    # Regression dataset for epsilon-SVR / nu-SVR.
    reg_set = "housing_scale"

    for kernel in (0, 1, 2, 3):
        for svm_type in (0, 1, 2):
            for ds in class_sets:
                cases.append(Case(svm_type=svm_type, kernel=kernel, dataset=ds))
        for svm_type in (3, 4):
            cases.append(Case(svm_type=svm_type, kernel=kernel, dataset=reg_set))

    class_pre = ["heart_scale.precomputed", "iris.scale.precomputed"]
    reg_pre = "housing_scale.precomputed"
    for svm_type in (0, 1, 2):
        for ds in class_pre:
            cases.append(Case(svm_type=svm_type, kernel=4, dataset=ds))
    for svm_type in (3, 4):
        cases.append(Case(svm_type=svm_type, kernel=4, dataset=reg_pre))

    return cases


def benchmark_case(
    case: Case,
    warmup: int,
    runs: int,
    rust_train: Path,
    rust_predict: Path,
    c_train: Path,
    c_predict: Path,
    data_dir: Path,
    tmp_dir: Path,
) -> Dict[str, object]:
    data_path = data_dir / case.dataset
    if not data_path.exists():
        raise RuntimeError(f"dataset not found: {data_path}")

    c_model = tmp_dir / f"c_{case.id}.model"
    r_model = tmp_dir / f"r_{case.id}.model"
    c_pred = tmp_dir / f"c_{case.id}.pred"
    r_pred = tmp_dir / f"r_{case.id}.pred"

    c_model_prob = tmp_dir / f"c_{case.id}.prob.model"
    r_model_prob = tmp_dir / f"r_{case.id}.prob.model"
    c_pred_prob = tmp_dir / f"c_{case.id}.prob.pred"
    r_pred_prob = tmp_dir / f"r_{case.id}.prob.pred"

    rust_train_cmd = [
        str(rust_train),
        "-q",
        "-s",
        str(case.svm_type),
        "-t",
        str(case.kernel),
        str(data_path),
        str(r_model),
    ]
    c_train_cmd = [
        str(c_train),
        "-q",
        "-s",
        str(case.svm_type),
        "-t",
        str(case.kernel),
        str(data_path),
        str(c_model),
    ]

    rust_predict_cmd = [
        str(rust_predict),
        "-q",
        str(data_path),
        str(r_model),
        str(r_pred),
    ]
    c_predict_cmd = [
        str(c_predict),
        "-q",
        str(data_path),
        str(c_model),
        str(c_pred),
    ]

    rust_train_samples = measure_command(rust_train_cmd, warmup, runs)
    c_train_samples = measure_command(c_train_cmd, warmup, runs)

    # Setup model once for pure prediction timings.
    run(rust_train_cmd)
    run(c_train_cmd)
    rust_predict_samples = measure_command(rust_predict_cmd, warmup, runs)
    c_predict_samples = measure_command(c_predict_cmd, warmup, runs)

    operations: Dict[str, Dict[str, object]] = {
        "train": {
            "rust": summary(rust_train_samples),
            "c": summary(c_train_samples),
        },
        "predict": {
            "rust": summary(rust_predict_samples),
            "c": summary(c_predict_samples),
        },
    }

    # Probability ops are valid for C-SVC, nu-SVC, epsilon-SVR, nu-SVR.
    if case.svm_type in (0, 1, 3, 4):
        rust_train_prob_cmd = [
            str(rust_train),
            "-q",
            "-b",
            "1",
            "-s",
            str(case.svm_type),
            "-t",
            str(case.kernel),
            str(data_path),
            str(r_model_prob),
        ]
        c_train_prob_cmd = [
            str(c_train),
            "-q",
            "-b",
            "1",
            "-s",
            str(case.svm_type),
            "-t",
            str(case.kernel),
            str(data_path),
            str(c_model_prob),
        ]

        rust_predict_prob_cmd = [
            str(rust_predict),
            "-q",
            "-b",
            "1",
            str(data_path),
            str(r_model_prob),
            str(r_pred_prob),
        ]
        c_predict_prob_cmd = [
            str(c_predict),
            "-q",
            "-b",
            "1",
            str(data_path),
            str(c_model_prob),
            str(c_pred_prob),
        ]

        rust_train_prob_samples = measure_command(rust_train_prob_cmd, warmup, runs)
        c_train_prob_samples = measure_command(c_train_prob_cmd, warmup, runs)

        run(rust_train_prob_cmd)
        run(c_train_prob_cmd)
        rust_predict_prob_samples = measure_command(rust_predict_prob_cmd, warmup, runs)
        c_predict_prob_samples = measure_command(c_predict_prob_cmd, warmup, runs)

        operations["train_probability"] = {
            "rust": summary(rust_train_prob_samples),
            "c": summary(c_train_prob_samples),
        }
        operations["predict_probability"] = {
            "rust": summary(rust_predict_prob_samples),
            "c": summary(c_predict_prob_samples),
        }

    for op_name, op_data in operations.items():
        rust_median = float(op_data["rust"]["median_ms"])
        c_median = float(op_data["c"]["median_ms"])
        op_data["rust_over_c_median_ratio"] = round(rust_median / c_median, 6)

    return {
        "case": {
            "id": case.id,
            "svm_type": case.svm_type,
            "kernel": case.kernel,
            "dataset": case.dataset,
        },
        "operations": operations,
    }


def aggregate(results: List[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    by_op: Dict[str, Dict[str, List[float]]] = {}
    for case_result in results:
        operations: Dict[str, Dict[str, object]] = case_result["operations"]  # type: ignore[assignment]
        for op_name, op_data in operations.items():
            entry = by_op.setdefault(op_name, {"rust": [], "c": [], "ratio": []})
            entry["rust"].append(float(op_data["rust"]["median_ms"]))  # type: ignore[index]
            entry["c"].append(float(op_data["c"]["median_ms"]))  # type: ignore[index]
            entry["ratio"].append(float(op_data["rust_over_c_median_ratio"]))

    out: Dict[str, Dict[str, float]] = {}
    for op_name, vals in by_op.items():
        out[op_name] = {
            "cases": float(len(vals["ratio"])),
            "rust_median_of_medians_ms": round(statistics.median(vals["rust"]), 6),
            "c_median_of_medians_ms": round(statistics.median(vals["c"]), 6),
            "rust_over_c_median_ratio": round(statistics.median(vals["ratio"]), 6),
            "rust_over_c_p95_ratio": round(percentile(sorted(vals["ratio"]), 95.0), 6),
            "rust_over_c_max_ratio": round(max(vals["ratio"]), 6),
        }
    return out


def write_report(
    results: List[Dict[str, object]],
    aggregate_stats: Dict[str, Dict[str, float]],
    report_path: Path,
    raw_json_path: Path,
    warmup: int,
    runs: int,
) -> None:
    lines: List[str] = []
    lines.append("# Benchmark Report")
    lines.append("")
    lines.append(f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}")
    lines.append("")
    lines.append("This report compares CLI performance of Rust (`svm-*-rs`) vs C (`vendor/libsvm`).")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(f"- Warmup runs per command: `{warmup}`")
    lines.append(f"- Measured runs per command: `{runs}`")
    lines.append("- Timing metric: wall clock (`perf_counter_ns`) per command invocation")
    lines.append("- Summary metric: per-case median and p95 from repeated runs")
    lines.append("")
    lines.append("## Aggregate Results")
    lines.append("")
    lines.append("| Operation | Cases | Rust median-of-medians (ms) | C median-of-medians (ms) | Rust/C median ratio | Rust/C p95 ratio | Worst-case ratio |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    for op_name in sorted(aggregate_stats.keys()):
        stats = aggregate_stats[op_name]
        lines.append(
            f"| `{op_name}` | {int(stats['cases'])} | {stats['rust_median_of_medians_ms']:.3f} | "
            f"{stats['c_median_of_medians_ms']:.3f} | {stats['rust_over_c_median_ratio']:.3f} | "
            f"{stats['rust_over_c_p95_ratio']:.3f} | {stats['rust_over_c_max_ratio']:.3f} |"
        )

    # Top 10 slowest Rust/C ratios across all operations.
    rank_rows: List[tuple[float, str, str]] = []
    for case_result in results:
        case = case_result["case"]  # type: ignore[assignment]
        operations = case_result["operations"]  # type: ignore[assignment]
        case_id = case["id"]  # type: ignore[index]
        for op_name, op_data in operations.items():
            ratio = float(op_data["rust_over_c_median_ratio"])  # type: ignore[index]
            rank_rows.append((ratio, case_id, op_name))
    rank_rows.sort(reverse=True, key=lambda x: x[0])

    lines.append("")
    lines.append("## Highest Rust/C Ratios")
    lines.append("")
    lines.append("| Case | Operation | Rust/C median ratio |")
    lines.append("|---|---|---:|")
    for ratio, case_id, op_name in rank_rows[:10]:
        lines.append(f"| `{case_id}` | `{op_name}` | {ratio:.3f} |")

    lines.append("")
    lines.append(f"Raw data: `reference/{raw_json_path.name}`")
    lines.append("")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    scripts_dir = root / "scripts"
    data_dir = root / "data"
    reference_dir = root / "reference"
    tmp_dir = root / ".tmp" / "benchmark_compare"
    reference_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    warmup = int(os.environ.get("BENCH_WARMUP", "1"))
    runs = int(os.environ.get("BENCH_RUNS", "3"))
    if warmup < 0 or runs < 1:
        raise RuntimeError("invalid BENCH_WARMUP/BENCH_RUNS")

    # Ensure precomputed datasets exist.
    run(["python3", str(scripts_dir / "generate_precomputed_datasets.py")], quiet=True)

    rust_train = root / "target" / "release" / "svm-train-rs"
    rust_predict = root / "target" / "release" / "svm-predict-rs"
    c_train = root / "vendor" / "libsvm" / "svm-train"
    c_predict = root / "vendor" / "libsvm" / "svm-predict"

    if not rust_train.exists() or not rust_predict.exists():
        run(
            [
                "cargo",
                "build",
                "--release",
                "-p",
                "svm-train-rs",
                "-p",
                "svm-predict-rs",
            ],
            quiet=False,
        )
    if not c_train.exists() or not c_predict.exists():
        run(["make", "-C", str(root / "vendor" / "libsvm")], quiet=False)

    cases = dataset_matrix()
    results: List[Dict[str, object]] = []

    for idx, case in enumerate(cases, start=1):
        print(
            f"[{idx:02d}/{len(cases)}] benchmarking {case.id} "
            f"(s={case.svm_type}, t={case.kernel}, data={case.dataset})"
        )
        case_result = benchmark_case(
            case=case,
            warmup=warmup,
            runs=runs,
            rust_train=rust_train,
            rust_predict=rust_predict,
            c_train=c_train,
            c_predict=c_predict,
            data_dir=data_dir,
            tmp_dir=tmp_dir,
        )
        results.append(case_result)

    aggregate_stats = aggregate(results)
    raw_json_path = reference_dir / "benchmark_results.json"
    report_path = reference_dir / "benchmark_report.md"

    payload = {
        "metadata": {
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "warmup_runs": warmup,
            "measured_runs": runs,
            "case_count": len(cases),
        },
        "aggregate": aggregate_stats,
        "cases": results,
    }
    raw_json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_report(results, aggregate_stats, report_path, raw_json_path, warmup, runs)

    print(f"Wrote {raw_json_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
