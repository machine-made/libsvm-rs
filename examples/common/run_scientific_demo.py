#!/usr/bin/env python3
"""Run scientific Rust-vs-C++ LIBSVM demo experiments and generate plots."""

from __future__ import annotations

import argparse
import bz2
import json
import lzma
import math
import os
import statistics
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from datasets_catalog import DATASETS


@dataclass(frozen=True)
class Paths:
    root: Path
    raw_dir: Path
    processed_dir: Path
    subsets_dir: Path
    option_dir: Path
    output_dir: Path
    tmp_dir: Path


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


def run(cmd: List[str], quiet: bool = True) -> None:
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

    out: List[float] = []
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        run(cmd, quiet=True)
        t1 = time.perf_counter_ns()
        out.append((t1 - t0) / 1e6)
    return out


def open_text_reader(path: Path):
    if path.suffix == ".bz2":
        return bz2.open(path, "rt", encoding="utf-8", errors="replace")
    if path.suffix == ".xz":
        return lzma.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    result = subprocess.run(
        [
            "curl",
            "-fL",
            "--retry",
            "3",
            "--retry-delay",
            "2",
            "-o",
            str(tmp),
            url,
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        raise RuntimeError(f"curl failed ({result.returncode}) for {url}")
    os.replace(tmp, dest)


def ensure_raw_dataset(paths: Paths, dataset_id: str) -> None:
    spec = DATASETS[dataset_id]
    urls = [u for u in [spec.train_url, spec.test_url, spec.single_url] if u]
    for url in urls:
        dest = paths.raw_dir / Path(url).name
        if dest.exists():
            continue
        print(f"Downloading missing raw file: {dest.name}")
        download(url, dest)


def stream_copy_lines(src: Path, dst: Path, max_lines: Optional[int] = None) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open_text_reader(src) as f_in, dst.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            f_out.write(line)
            count += 1
            if max_lines is not None and count >= max_lines:
                break
    return count


def split_single_file(src: Path, train_dst: Path, test_dst: Path, train_rows: int, test_rows: Optional[int]) -> Tuple[int, int]:
    train_dst.parent.mkdir(parents=True, exist_ok=True)
    test_dst.parent.mkdir(parents=True, exist_ok=True)
    n_train = 0
    n_test = 0
    with open_text_reader(src) as f_in, train_dst.open("w", encoding="utf-8") as f_train, test_dst.open("w", encoding="utf-8") as f_test:
        for i, line in enumerate(f_in):
            if i < train_rows:
                f_train.write(line)
                n_train += 1
                continue
            if test_rows is None or n_test < test_rows:
                f_test.write(line)
                n_test += 1
                continue
            break
    return n_train, n_test


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def ensure_processed_dataset(paths: Paths, dataset_id: str) -> Tuple[Path, Path]:
    spec = DATASETS[dataset_id]
    train_out = paths.processed_dir / f"{dataset_id}.train"
    test_out = paths.processed_dir / f"{dataset_id}.test"

    if train_out.exists() and test_out.exists():
        return train_out, test_out

    ensure_raw_dataset(paths, dataset_id)

    if spec.needs_split:
        assert spec.single_url is not None
        raw = paths.raw_dir / Path(spec.single_url).name
        train_rows = spec.train_rows
        if train_rows is None:
            raise RuntimeError(f"missing train_rows for split dataset {dataset_id}")
        n_train, n_test = split_single_file(
            raw,
            train_out,
            test_out,
            train_rows=train_rows,
            test_rows=spec.test_rows,
        )
        print(f"Prepared {dataset_id}: train={n_train} test={n_test}")
    else:
        if spec.train_url is None or spec.test_url is None:
            raise RuntimeError(f"dataset {dataset_id} missing urls")
        raw_train = paths.raw_dir / Path(spec.train_url).name
        raw_test = paths.raw_dir / Path(spec.test_url).name
        n_train = stream_copy_lines(raw_train, train_out)
        n_test = stream_copy_lines(raw_test, test_out)
        print(f"Prepared {dataset_id}: train={n_train} test={n_test}")

    return train_out, test_out


def ensure_subset(src: Path, dataset_id: str, subset_kind: str, size: int, subsets_dir: Path) -> Path:
    out = subsets_dir / dataset_id / f"{subset_kind}_{size}.svm"
    if out.exists():
        return out

    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with src.open("r", encoding="utf-8") as f_in, out.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            f_out.write(line)
            count += 1
            if count >= size:
                break
    if count < size:
        raise RuntimeError(f"requested {size} lines from {src}, only got {count}")
    return out


def read_labels(libsvm_file: Path) -> List[float]:
    out: List[float] = []
    with libsvm_file.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            token = s.split()[0]
            out.append(float(token))
    return out


def read_predictions(path: Path) -> List[float]:
    out: List[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            out.append(float(s.split()[0]))
    return out


def calc_metrics(task: str, labels: List[float], rust_preds: List[float], c_preds: List[float]) -> Dict[str, float]:
    if len(labels) != len(rust_preds) or len(labels) != len(c_preds):
        raise RuntimeError("label/pred length mismatch")

    n = len(labels)
    if n == 0:
        raise RuntimeError("no labels to evaluate")

    if task == "classification":
        rust_ok = sum(1 for y, p in zip(labels, rust_preds) if y == p)
        c_ok = sum(1 for y, p in zip(labels, c_preds) if y == p)
        agree = sum(1 for pr, pc in zip(rust_preds, c_preds) if pr == pc)
        return {
            "rust_accuracy": rust_ok / n,
            "c_accuracy": c_ok / n,
            "prediction_agreement": agree / n,
        }

    # regression
    rust_sq = 0.0
    c_sq = 0.0
    diff_abs = 0.0
    diff_max = 0.0
    for y, pr, pc in zip(labels, rust_preds, c_preds):
        dr = pr - y
        dc = pc - y
        rust_sq += dr * dr
        c_sq += dc * dc
        dd = abs(pr - pc)
        diff_abs += dd
        if dd > diff_max:
            diff_max = dd
    return {
        "rust_rmse": math.sqrt(rust_sq / n),
        "c_rmse": math.sqrt(c_sq / n),
        "rust_c_mae": diff_abs / n,
        "rust_c_max_abs": diff_max,
    }


def ensure_binaries(root: Path) -> Tuple[Path, Path, Path, Path]:
    rust_train = root / "target" / "release" / "svm-train-rs"
    rust_predict = root / "target" / "release" / "svm-predict-rs"
    c_train = root / "vendor" / "libsvm" / "svm-train"
    c_predict = root / "vendor" / "libsvm" / "svm-predict"

    if not rust_train.exists() or not rust_predict.exists():
        print("Building Rust release binaries...")
        run(["cargo", "build", "--release", "-p", "svm-train-rs", "-p", "svm-predict-rs"], quiet=False)

    if not c_train.exists() or not c_predict.exists():
        print("Building C reference binaries...")
        run(["make", "-C", str(root / "vendor" / "libsvm")], quiet=False)

    return rust_train, rust_predict, c_train, c_predict


def build_train_cmd(binary: Path, svm_type: int, kernel: int, c_value: float, train_file: Path, model_file: Path, gamma: Optional[float]) -> List[str]:
    cmd = [
        str(binary),
        "-q",
        "-s",
        str(svm_type),
        "-t",
        str(kernel),
        "-c",
        str(c_value),
    ]
    if gamma is not None:
        cmd.extend(["-g", f"{gamma:.17g}"])
    cmd.extend([str(train_file), str(model_file)])
    return cmd


def build_predict_cmd(binary: Path, test_file: Path, model_file: Path, pred_file: Path) -> List[str]:
    return [str(binary), "-q", str(test_file), str(model_file), str(pred_file)]


def kernel_name(kernel: int) -> str:
    return {
        0: "linear",
        1: "poly",
        2: "rbf",
        3: "sigmoid",
        4: "precomputed",
    }.get(kernel, f"k{kernel}")


def sanitize_float(v: float) -> str:
    return f"{v:.6g}".replace("-", "m").replace(".", "p")


def safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def format_class_label(v: float) -> str:
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:g}"


def build_confusion_counts(labels: List[float], preds: List[float], classes: List[float]) -> np.ndarray:
    idx = {label: i for i, label in enumerate(classes)}
    mat = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for y_true, y_pred in zip(labels, preds):
        mat[idx[y_true], idx[y_pred]] += 1
    return mat


def select_plot_sample(n: int, max_points: int = 12000) -> np.ndarray:
    if n <= max_points:
        return np.arange(n, dtype=np.int64)
    step = max(1, n // max_points)
    out = np.arange(0, n, step, dtype=np.int64)
    return out[:max_points]


def plot_confusion_matrices(
    dataset_id: str,
    labels: List[float],
    rust_preds: List[float],
    c_preds: List[float],
    out_png: Path,
    kernel: int,
    c_value: float,
) -> None:
    classes = sorted(set(labels) | set(rust_preds) | set(c_preds))
    class_names = [format_class_label(v) for v in classes]
    cm_rust = build_confusion_counts(labels, rust_preds, classes)
    cm_c = build_confusion_counts(labels, c_preds, classes)
    cm_delta = cm_rust.astype(np.int64) - cm_c.astype(np.int64)
    vmax = int(max(np.max(cm_rust), np.max(cm_c), 1))
    delta_abs = int(max(abs(int(np.min(cm_delta))), abs(int(np.max(cm_delta))), 1))
    agreement = sum(1 for pr, pc in zip(rust_preds, c_preds) if pr == pc) / len(labels)

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.2))
    mats = [
        (cm_rust, "Rust confusion", "Blues", 0, vmax),
        (cm_c, "C++ confusion", "Blues", 0, vmax),
        (cm_delta, "Rust - C++", "coolwarm", -delta_abs, delta_abs),
    ]
    for ax, (mat, title, cmap, vmin, vmax_local) in zip(axes, mats):
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax_local, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_yticklabels(class_names)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        if len(class_names) <= 10:
            threshold = (vmax_local + vmin) / 2.0
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    val = int(mat[i, j])
                    color = "white" if val > threshold else "black"
                    ax.text(j, i, str(val), ha="center", va="center", fontsize=8, color=color)

    fig.suptitle(
        f"{dataset_id} baseline confusion (kernel={kernel_name(kernel)}, C={c_value:g}, "
        f"n={len(labels)}, Rust/C++ agreement={agreement:.4f})"
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_regression_residuals(
    dataset_id: str,
    labels: List[float],
    rust_preds: List[float],
    c_preds: List[float],
    out_png: Path,
    kernel: int,
    c_value: float,
) -> None:
    y_true = np.asarray(labels, dtype=float)
    y_rust = np.asarray(rust_preds, dtype=float)
    y_c = np.asarray(c_preds, dtype=float)
    res_rust = y_rust - y_true
    res_c = y_c - y_true
    delta = y_rust - y_c
    idx = select_plot_sample(len(labels), max_points=12000)

    lo = float(min(np.min(y_true), np.min(y_rust), np.min(y_c)))
    hi = float(max(np.max(y_true), np.max(y_rust), np.max(y_c)))
    mae_delta = float(np.mean(np.abs(delta)))
    max_abs_delta = float(np.max(np.abs(delta)))

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.0))

    ax = axes[0, 0]
    ax.scatter(y_true[idx], y_rust[idx], s=6, alpha=0.25, edgecolors="none")
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    ax.set_title("Rust prediction vs true")
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.grid(True, alpha=0.25)

    ax = axes[0, 1]
    ax.scatter(y_true[idx], y_c[idx], s=6, alpha=0.25, edgecolors="none")
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    ax.set_title("C++ prediction vs true")
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    ax.hist(res_rust, bins=80, alpha=0.6, label="Rust residual", density=True)
    ax.hist(res_c, bins=80, alpha=0.6, label="C++ residual", density=True)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Residual distributions")
    ax.set_xlabel("Prediction - true")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1, 1]
    ax.scatter(y_true[idx], delta[idx], s=6, alpha=0.25, edgecolors="none", label="Rust - C++")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Rust-C++ delta vs true")
    ax.set_xlabel("True")
    ax.set_ylabel("Rust - C++")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.suptitle(
        f"{dataset_id} baseline residual diagnostics (kernel={kernel_name(kernel)}, C={c_value:g}, "
        f"n={len(labels)}, MAE={mae_delta:.4e}, max|delta|={max_abs_delta:.4e})"
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def select_baseline_case(rows: List[Dict[str, object]], baseline_kernel: int, baseline_c: float) -> Optional[Dict[str, object]]:
    baseline_rows = [
        r
        for r in rows
        if int(r["kernel"]) == baseline_kernel and abs(float(r["c_value"]) - baseline_c) < 1e-12
    ]
    if not baseline_rows:
        return None
    baseline_rows.sort(key=lambda r: int(r["train_rows"]))
    return baseline_rows[-1]


def run_one_case(
    dataset_id: str,
    task: str,
    labels: List[float],
    train_file: Path,
    train_rows: int,
    test_file: Path,
    svm_type: int,
    kernel: int,
    c_value: float,
    gamma: Optional[float],
    warmup: int,
    runs: int,
    rust_train: Path,
    rust_predict: Path,
    c_train: Path,
    c_predict: Path,
    tmp_dir: Path,
    keep_artifacts: bool = False,
) -> Dict[str, object]:
    case_tag = f"{dataset_id}_s{svm_type}_k{kernel}_c{sanitize_float(c_value)}_n{train_rows}"

    r_model = tmp_dir / f"rust_{case_tag}.model"
    c_model = tmp_dir / f"c_{case_tag}.model"
    r_pred = tmp_dir / f"rust_{case_tag}.pred"
    c_pred = tmp_dir / f"c_{case_tag}.pred"

    rust_train_cmd = build_train_cmd(rust_train, svm_type, kernel, c_value, train_file, r_model, gamma)
    c_train_cmd = build_train_cmd(c_train, svm_type, kernel, c_value, train_file, c_model, gamma)
    rust_predict_cmd = build_predict_cmd(rust_predict, test_file, r_model, r_pred)
    c_predict_cmd = build_predict_cmd(c_predict, test_file, c_model, c_pred)

    rust_train_samples = measure_command(rust_train_cmd, warmup, runs)
    c_train_samples = measure_command(c_train_cmd, warmup, runs)

    run(rust_train_cmd)
    run(c_train_cmd)

    rust_predict_samples = measure_command(rust_predict_cmd, warmup, runs)
    c_predict_samples = measure_command(c_predict_cmd, warmup, runs)

    run(rust_predict_cmd)
    run(c_predict_cmd)

    rust_preds = read_predictions(r_pred)
    c_preds = read_predictions(c_pred)
    metrics = calc_metrics(task, labels, rust_preds, c_preds)

    train_med_rust = float(statistics.median(rust_train_samples))
    train_med_c = float(statistics.median(c_train_samples))
    pred_med_rust = float(statistics.median(rust_predict_samples))
    pred_med_c = float(statistics.median(c_predict_samples))

    out: Dict[str, object] = {
        "dataset": dataset_id,
        "task": task,
        "train_rows": train_rows,
        "test_rows": len(labels),
        "svm_type": svm_type,
        "kernel": kernel,
        "kernel_name": kernel_name(kernel),
        "c_value": c_value,
        "gamma": gamma,
        "timing": {
            "train": {
                "rust": summary(rust_train_samples),
                "c": summary(c_train_samples),
                "rust_over_c_median_ratio": round(train_med_rust / train_med_c, 6),
            },
            "predict": {
                "rust": summary(rust_predict_samples),
                "c": summary(c_predict_samples),
                "rust_over_c_median_ratio": round(pred_med_rust / pred_med_c, 6),
            },
        },
        "metrics": metrics,
    }

    if keep_artifacts:
        out["artifacts"] = {
            "rust_model": str(r_model),
            "c_model": str(c_model),
            "rust_pred": str(r_pred),
            "c_pred": str(c_pred),
            "test_file": str(test_file),
        }
    else:
        safe_unlink(r_model)
        safe_unlink(c_model)
        safe_unlink(r_pred)
        safe_unlink(c_pred)

    return out


def write_report(option_title: str, config: Dict[str, object], results: List[Dict[str, object]], out_md: Path) -> None:
    lines: List[str] = []
    lines.append(f"# {option_title} Results")
    lines.append("")
    lines.append(f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Datasets: {', '.join(config['datasets'])}")
    lines.append(f"- Kernels: {config['kernels']}")
    lines.append(f"- C values: {config['c_values']}")
    lines.append(f"- Warmup: {config['warmup']}")
    lines.append(f"- Runs: {config['runs']}")
    lines.append("")
    lines.append("## Cases")
    lines.append("")
    lines.append("| Dataset | Kernel | C | Train rows | Test rows | Train Rust/C ratio | Predict Rust/C ratio | Metric summary |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")

    for row in results:
        dataset = row["dataset"]
        kernel = row["kernel_name"]
        c_value = row["c_value"]
        train_rows = row["train_rows"]
        test_rows = row["test_rows"]
        tr_ratio = row["timing"]["train"]["rust_over_c_median_ratio"]
        pr_ratio = row["timing"]["predict"]["rust_over_c_median_ratio"]
        metrics = row["metrics"]
        if row["task"] == "classification":
            metric_txt = (
                f"acc_r={metrics['rust_accuracy']:.4f}, "
                f"acc_c={metrics['c_accuracy']:.4f}, "
                f"agree={metrics['prediction_agreement']:.4f}"
            )
        else:
            metric_txt = (
                f"rmse_r={metrics['rust_rmse']:.4f}, "
                f"rmse_c={metrics['c_rmse']:.4f}, "
                f"mae_rc={metrics['rust_c_mae']:.4e}"
            )
        lines.append(
            f"| `{dataset}` | `{kernel}` | {c_value:.3g} | {train_rows} | {test_rows} | {tr_ratio:.3f} | {pr_ratio:.3f} | {metric_txt} |"
        )

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_terminal_summary(results: List[Dict[str, object]]) -> None:
    if not results:
        return

    print("\n=== Quick Timing Summary (median Rust/C++) ===")
    print(f"{'Dataset':<22} {'Kernel':<8} {'TrainRatio':>10} {'PredRatio':>10}")
    print("-" * 56)
    for row in results:
        ds = str(row["dataset"])
        k = str(row["kernel_name"])
        tr = float(row["timing"]["train"]["rust_over_c_median_ratio"])
        pr = float(row["timing"]["predict"]["rust_over_c_median_ratio"])
        print(f"{ds:<22} {k:<8} {tr:>10.3f} {pr:>10.3f}")


def plot_dataset_curves(dataset_id: str, rows: List[Dict[str, object]], out_png: Path, baseline_kernel: int, baseline_c: float) -> None:
    baseline = [
        r
        for r in rows
        if int(r["kernel"]) == baseline_kernel and abs(float(r["c_value"]) - baseline_c) < 1e-12
    ]
    if not baseline:
        return
    baseline.sort(key=lambda r: int(r["train_rows"]))

    xs = [int(r["train_rows"]) for r in baseline]
    train_r = [float(r["timing"]["train"]["rust"]["median_ms"]) for r in baseline]
    train_c = [float(r["timing"]["train"]["c"]["median_ms"]) for r in baseline]
    pred_r = [float(r["timing"]["predict"]["rust"]["median_ms"]) for r in baseline]
    pred_c = [float(r["timing"]["predict"]["c"]["median_ms"]) for r in baseline]
    ratio = [float(r["timing"]["train"]["rust_over_c_median_ratio"]) for r in baseline]

    task = str(baseline[0]["task"])

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

    ax = axes[0, 0]
    ax.plot(xs, train_r, marker="o", label="Rust")
    ax.plot(xs, train_c, marker="o", label="C++")
    ax.set_title("Training Time vs Train Size")
    ax.set_xlabel("Train rows")
    ax.set_ylabel("Median ms")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(xs, pred_r, marker="o", label="Rust")
    ax.plot(xs, pred_c, marker="o", label="C++")
    ax.set_title("Prediction Time vs Train Size")
    ax.set_xlabel("Train rows")
    ax.set_ylabel("Median ms")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(xs, ratio, marker="o", label="Train Rust/C++")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Speed Ratio (Train)")
    ax.set_xlabel("Train rows")
    ax.set_ylabel("Rust/C++")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    if task == "classification":
        acc_r = [float(r["metrics"]["rust_accuracy"]) for r in baseline]
        acc_c = [float(r["metrics"]["c_accuracy"]) for r in baseline]
        agree = [float(r["metrics"]["prediction_agreement"]) for r in baseline]
        ax.plot(xs, acc_r, marker="o", label="Rust accuracy")
        ax.plot(xs, acc_c, marker="o", label="C++ accuracy")
        ax.plot(xs, agree, marker="o", label="Prediction agreement")
        ax.set_ylim(0.0, 1.01)
        ax.set_ylabel("Fraction")
        ax.set_title("Correctness / Parity")
    else:
        rmse_r = [float(r["metrics"]["rust_rmse"]) for r in baseline]
        rmse_c = [float(r["metrics"]["c_rmse"]) for r in baseline]
        mae = [float(r["metrics"]["rust_c_mae"]) for r in baseline]
        ax.plot(xs, rmse_r, marker="o", label="Rust RMSE")
        ax.plot(xs, rmse_c, marker="o", label="C++ RMSE")
        ax.plot(xs, mae, marker="o", label="Rust-C++ MAE")
        ax.set_ylabel("Error")
        ax.set_title("Regression Metrics")

    ax.set_xlabel("Train rows")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle(f"{dataset_id} (kernel={kernel_name(baseline_kernel)}, C={baseline_c:g})")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to option config JSON")
    parser.add_argument("--download-only", action="store_true", help="Only download and prepare datasets")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    option_dir = config_path.parent
    root = config_path.parents[2]

    config = json.loads(config_path.read_text(encoding="utf-8"))

    paths = Paths(
        root=root,
        raw_dir=root / "examples" / "data" / "raw",
        processed_dir=root / "examples" / "data" / "processed",
        subsets_dir=root / "examples" / "data" / "subsets",
        option_dir=option_dir,
        output_dir=option_dir / "output",
        tmp_dir=option_dir / "output" / "tmp",
    )
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    paths.subsets_dir.mkdir(parents=True, exist_ok=True)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    paths.tmp_dir.mkdir(parents=True, exist_ok=True)

    datasets: List[str] = config["datasets"]
    for dataset_id in datasets:
        if dataset_id not in DATASETS:
            raise RuntimeError(f"Unknown dataset: {dataset_id}")

    prepared: Dict[str, Tuple[Path, Path]] = {}
    for dataset_id in datasets:
        prepared[dataset_id] = ensure_processed_dataset(paths, dataset_id)

    if args.download_only:
        print("Download and preprocessing completed.")
        return

    rust_train, rust_predict, c_train, c_predict = ensure_binaries(root)

    runs = int(config.get("runs", 3))
    warmup = int(config.get("warmup", 1))
    kernels: List[int] = [int(k) for k in config.get("kernels", [0, 2])]
    c_values: List[float] = [float(v) for v in config.get("c_values", [1.0])]
    baseline_kernel = int(kernels[0])
    baseline_c = float(c_values[0])
    train_sizes_cfg: Dict[str, List[int]] = {
        k: [int(x) for x in vals] for k, vals in config.get("train_sizes", {}).items()
    }
    test_rows_cfg: Dict[str, int] = {k: int(v) for k, v in config.get("test_rows", {}).items()}

    results: List[Dict[str, object]] = []

    for dataset_id in datasets:
        spec = DATASETS[dataset_id]
        train_full, test_full = prepared[dataset_id]

        full_train_rows = count_lines(train_full)
        full_test_rows = count_lines(test_full)

        train_sizes = train_sizes_cfg.get(dataset_id, [min(2000, full_train_rows), min(10000, full_train_rows), full_train_rows])
        train_sizes = sorted({min(s, full_train_rows) for s in train_sizes if s > 0})

        test_rows = min(test_rows_cfg.get(dataset_id, full_test_rows), full_test_rows)
        if test_rows < full_test_rows:
            test_file = ensure_subset(test_full, dataset_id, "test", test_rows, paths.subsets_dir)
        else:
            test_file = test_full

        labels = read_labels(test_file)

        for size in train_sizes:
            if size < full_train_rows:
                train_file = ensure_subset(train_full, dataset_id, "train", size, paths.subsets_dir)
            else:
                train_file = train_full

            for kernel in kernels:
                for c_value in c_values:
                    gamma = (1.0 / spec.features) if kernel == 2 else None
                    print(
                        f"Running {dataset_id}: train={size} test={len(labels)} "
                        f"kernel={kernel_name(kernel)} C={c_value:g}"
                    )
                    row = run_one_case(
                        dataset_id=dataset_id,
                        task=spec.task,
                        labels=labels,
                        train_file=train_file,
                        train_rows=size,
                        test_file=test_file,
                        svm_type=spec.svm_type,
                        kernel=kernel,
                        c_value=c_value,
                        gamma=gamma,
                        warmup=warmup,
                        runs=runs,
                        rust_train=rust_train,
                        rust_predict=rust_predict,
                        c_train=c_train,
                        c_predict=c_predict,
                        tmp_dir=paths.tmp_dir,
                        keep_artifacts=(kernel == baseline_kernel and abs(c_value - baseline_c) < 1e-12),
                    )
                    results.append(row)

    out_json = paths.output_dir / "results.json"
    out_json.write_text(
        json.dumps(
            {
                "metadata": {
                    "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "config": config,
                },
                "results": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # Cache minimal plotting arrays to NPZ.
    npz_rows: List[Tuple[float, float, float, float, float]] = []
    for r in results:
        metric = r["metrics"]
        if r["task"] == "classification":
            metric_value = float(metric["prediction_agreement"])
        else:
            metric_value = float(metric["rust_c_mae"])
        npz_rows.append(
            (
                float(r["train_rows"]),
                float(r["timing"]["train"]["rust"]["median_ms"]),
                float(r["timing"]["train"]["c"]["median_ms"]),
                float(r["timing"]["predict"]["rust"]["median_ms"]),
                metric_value,
            )
        )

    npz = np.array(npz_rows, dtype=float)
    npz_path = paths.output_dir / "scientific_demo.npz"
    np.savez_compressed(npz_path, rows=npz)

    for dataset_id in datasets:
        ds_rows = [r for r in results if r["dataset"] == dataset_id]
        if not ds_rows:
            continue
        plot_dataset_curves(
            dataset_id,
            ds_rows,
            out_png=paths.output_dir / f"{dataset_id}.png",
            baseline_kernel=baseline_kernel,
            baseline_c=baseline_c,
        )
        baseline_row = select_baseline_case(ds_rows, baseline_kernel=baseline_kernel, baseline_c=baseline_c)
        if baseline_row is None:
            continue
        artifacts = baseline_row.get("artifacts")
        if not isinstance(artifacts, dict):
            continue
        rust_pred_path = artifacts.get("rust_pred")
        c_pred_path = artifacts.get("c_pred")
        test_file_path = artifacts.get("test_file")
        if not isinstance(rust_pred_path, str) or not isinstance(c_pred_path, str) or not isinstance(test_file_path, str):
            continue
        labels = read_labels(Path(test_file_path))
        rust_preds = read_predictions(Path(rust_pred_path))
        c_preds = read_predictions(Path(c_pred_path))
        if str(baseline_row["task"]) == "classification":
            out_diag = paths.output_dir / f"{dataset_id}_confusion.png"
            plot_confusion_matrices(
                dataset_id=dataset_id,
                labels=labels,
                rust_preds=rust_preds,
                c_preds=c_preds,
                out_png=out_diag,
                kernel=int(baseline_row["kernel"]),
                c_value=float(baseline_row["c_value"]),
            )
            print(f"Wrote {out_diag}")
        else:
            out_diag = paths.output_dir / f"{dataset_id}_residuals.png"
            plot_regression_residuals(
                dataset_id=dataset_id,
                labels=labels,
                rust_preds=rust_preds,
                c_preds=c_preds,
                out_png=out_diag,
                kernel=int(baseline_row["kernel"]),
                c_value=float(baseline_row["c_value"]),
            )
            print(f"Wrote {out_diag}")

    report_path = paths.output_dir / "report.md"
    write_report(str(config.get("title", config.get("option_id", "Scientific Demo"))), config, results, report_path)
    print_terminal_summary(results)

    # Update global cross-option timing figure.
    comparison_script = root / "examples" / "common" / "make_comparison_figure.py"
    subprocess.run(
        ["python3", str(comparison_script), "--root", str(root), "--out", "examples/comparison.png"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    print(f"Wrote {out_json}")
    print(f"Wrote {npz_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
