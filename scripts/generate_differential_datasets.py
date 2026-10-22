#!/usr/bin/env python3
"""
Generate deterministic synthetic datasets for Rust-vs-C differential testing.

Outputs:
- data/generated/*.scale
- data/generated/*.scale.precomputed
- reference/dataset_manifest.json
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class DatasetSpec:
    name: str
    task: str  # classification | regression | oneclass
    description: str
    seed: int
    n_samples: int
    n_features: int
    density: float


def dot_sparse_dense(x: Dict[int, float], w: List[float]) -> float:
    total = 0.0
    for i, v in x.items():
        total += v * w[i - 1]
    return total


def sample_sparse_vector(
    rng: random.Random, n_features: int, density: float, scale: float = 1.0
) -> Dict[int, float]:
    x: Dict[int, float] = {}
    for i in range(1, n_features + 1):
        if rng.random() < density:
            x[i] = rng.uniform(-1.0, 1.0) * scale
    if not x:
        # Keep every sample non-empty.
        idx = rng.randint(1, n_features)
        x[idx] = rng.uniform(-1.0, 1.0) * scale
    return x


def gen_binary_sparse(spec: DatasetSpec) -> Tuple[List[float], List[Dict[int, float]]]:
    rng = random.Random(spec.seed)
    w = [rng.uniform(-1.0, 1.0) for _ in range(spec.n_features)]
    bias = rng.uniform(-0.2, 0.2)
    labels: List[float] = []
    rows: List[Dict[int, float]] = []
    for _ in range(spec.n_samples):
        x = sample_sparse_vector(rng, spec.n_features, spec.density)
        score = dot_sparse_dense(x, w) + bias
        y = 1.0 if score >= 0.0 else -1.0
        if rng.random() < 0.08:
            y = -y
        labels.append(y)
        rows.append(x)
    return labels, rows


def gen_multiclass_dense(spec: DatasetSpec) -> Tuple[List[float], List[Dict[int, float]]]:
    rng = random.Random(spec.seed)
    n_classes = 3
    centroids: List[List[float]] = []
    for _ in range(n_classes):
        centroids.append([rng.uniform(-1.5, 1.5) for _ in range(spec.n_features)])

    labels: List[float] = []
    rows: List[Dict[int, float]] = []
    n_per = spec.n_samples // n_classes
    for c in range(n_classes):
        for _ in range(n_per):
            x: Dict[int, float] = {}
            for i in range(1, spec.n_features + 1):
                v = centroids[c][i - 1] + rng.gauss(0.0, 0.35)
                # Keep moderate sparsity by dropping near-zero values.
                if abs(v) > 0.04:
                    x[i] = v
            if not x:
                x[rng.randint(1, spec.n_features)] = rng.uniform(-0.1, 0.1)
            labels.append(float(c + 1))
            rows.append(x)
    return labels, rows


def gen_binary_imbalanced(spec: DatasetSpec) -> Tuple[List[float], List[Dict[int, float]]]:
    rng = random.Random(spec.seed)
    pos_count = max(8, int(spec.n_samples * 0.12))
    neg_count = spec.n_samples - pos_count
    labels: List[float] = []
    rows: List[Dict[int, float]] = []

    for _ in range(neg_count):
        x = sample_sparse_vector(rng, spec.n_features, spec.density)
        for k in list(x.keys()):
            x[k] += rng.uniform(-0.2, 0.0)
        labels.append(-1.0)
        rows.append(x)

    for _ in range(pos_count):
        x = sample_sparse_vector(rng, spec.n_features, spec.density)
        for k in list(x.keys()):
            x[k] += rng.uniform(0.0, 0.2)
        labels.append(1.0)
        rows.append(x)

    # Deterministic shuffle to avoid ordered labels.
    order = list(range(spec.n_samples))
    rng.shuffle(order)
    labels = [labels[i] for i in order]
    rows = [rows[i] for i in order]
    return labels, rows


def gen_regression_sparse(spec: DatasetSpec) -> Tuple[List[float], List[Dict[int, float]]]:
    rng = random.Random(spec.seed)
    w = [rng.uniform(-2.0, 2.0) for _ in range(spec.n_features)]
    labels: List[float] = []
    rows: List[Dict[int, float]] = []
    for _ in range(spec.n_samples):
        x = sample_sparse_vector(rng, spec.n_features, spec.density, scale=1.2)
        y = dot_sparse_dense(x, w) + rng.gauss(0.0, 0.25)
        labels.append(y)
        rows.append(x)
    return labels, rows


def gen_extreme_scale(spec: DatasetSpec) -> Tuple[List[float], List[Dict[int, float]]]:
    rng = random.Random(spec.seed)
    labels: List[float] = []
    rows: List[Dict[int, float]] = []
    for _ in range(spec.n_samples):
        x: Dict[int, float] = {}
        for i in range(1, spec.n_features + 1):
            if rng.random() < spec.density:
                sign = -1.0 if rng.random() < 0.5 else 1.0
                # Mix tiny and huge magnitudes deterministically.
                exp = rng.uniform(-8.0, 6.0)
                x[i] = sign * (10.0**exp)
        if not x:
            x[rng.randint(1, spec.n_features)] = 1e-6
        score = sum(math.copysign(1.0, v) for v in x.values())
        labels.append(1.0 if score >= 0 else -1.0)
        rows.append(x)
    return labels, rows


def gen_oneclass_cluster(spec: DatasetSpec) -> Tuple[List[float], List[Dict[int, float]]]:
    rng = random.Random(spec.seed)
    labels: List[float] = []
    rows: List[Dict[int, float]] = []
    centroid = [rng.uniform(-0.4, 0.4) for _ in range(spec.n_features)]
    for _ in range(spec.n_samples):
        x: Dict[int, float] = {}
        for i in range(1, spec.n_features + 1):
            if rng.random() < spec.density:
                x[i] = centroid[i - 1] + rng.gauss(0.0, 0.08)
        if not x:
            x[rng.randint(1, spec.n_features)] = rng.gauss(0.0, 0.08)
        labels.append(1.0)
        rows.append(x)
    return labels, rows


def write_libsvm(path: Path, labels: List[float], rows: List[Dict[int, float]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for y, x in zip(labels, rows):
            parts = [f"{y:.17g}"]
            for idx in sorted(x.keys()):
                parts.append(f"{idx}:{x[idx]:.17g}")
            f.write(" ".join(parts))
            f.write("\n")


def sparse_dot(a: Dict[int, float], b: Dict[int, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    total = 0.0
    for k, va in a.items():
        vb = b.get(k)
        if vb is not None:
            total += va * vb
    return total


def write_precomputed(base_path: Path, labels: List[float], rows: List[Dict[int, float]]) -> Path:
    n = len(rows)
    out_path = base_path.with_suffix(base_path.suffix + ".precomputed")
    kmat: List[List[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        kmat[i][i] = sparse_dot(rows[i], rows[i])
        for j in range(i + 1, n):
            v = sparse_dot(rows[i], rows[j])
            kmat[i][j] = v
            kmat[j][i] = v

    with out_path.open("w", encoding="utf-8") as out:
        for i, y in enumerate(labels, start=1):
            parts = [f"{y:.17g}", f"0:{i}"]
            parts.extend(f"{j}:{kmat[i - 1][j - 1]:.17g}" for j in range(1, n + 1))
            out.write(" ".join(parts))
            out.write("\n")
    return out_path


def stats(rows: List[Dict[int, float]], n_features: int) -> Dict[str, float]:
    nonzero = sum(len(r) for r in rows)
    total = len(rows) * n_features
    density = (nonzero / total) if total else 0.0
    return {
        "samples": len(rows),
        "features": n_features,
        "nonzero": nonzero,
        "density": density,
    }


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data" / "generated"
    ref_dir = root / "reference"
    data_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        DatasetSpec(
            name="gen_binary_sparse.scale",
            task="classification",
            description="Binary sparse classification with label noise.",
            seed=101,
            n_samples=120,
            n_features=48,
            density=0.15,
        ),
        DatasetSpec(
            name="gen_multiclass_dense.scale",
            task="classification",
            description="Three-class dense-ish classification.",
            seed=202,
            n_samples=150,
            n_features=24,
            density=0.85,
        ),
        DatasetSpec(
            name="gen_binary_imbalanced.scale",
            task="classification",
            description="Binary classification with ~12% positives.",
            seed=303,
            n_samples=140,
            n_features=40,
            density=0.20,
        ),
        DatasetSpec(
            name="gen_regression_sparse.scale",
            task="regression",
            description="Sparse linear-ish regression with Gaussian noise.",
            seed=404,
            n_samples=160,
            n_features=56,
            density=0.18,
        ),
        DatasetSpec(
            name="gen_extreme_scale.scale",
            task="classification",
            description="Classification with mixed tiny/huge feature magnitudes.",
            seed=505,
            n_samples=110,
            n_features=30,
            density=0.22,
        ),
        DatasetSpec(
            name="gen_oneclass_cluster.scale",
            task="oneclass",
            description="Single-cluster one-class style dataset (all labels +1).",
            seed=606,
            n_samples=130,
            n_features=20,
            density=0.35,
        ),
    ]

    generators = {
        "gen_binary_sparse.scale": gen_binary_sparse,
        "gen_multiclass_dense.scale": gen_multiclass_dense,
        "gen_binary_imbalanced.scale": gen_binary_imbalanced,
        "gen_regression_sparse.scale": gen_regression_sparse,
        "gen_extreme_scale.scale": gen_extreme_scale,
        "gen_oneclass_cluster.scale": gen_oneclass_cluster,
    }

    manifest_entries: List[Dict[str, object]] = []

    for spec in specs:
        labels, rows = generators[spec.name](spec)
        out_path = data_dir / spec.name
        write_libsvm(out_path, labels, rows)
        pre_path = write_precomputed(out_path, labels, rows)

        st = stats(rows, spec.n_features)
        entry = {
            "name": spec.name,
            "task": spec.task,
            "description": spec.description,
            "seed": spec.seed,
            "path": str(out_path.relative_to(root)),
            "precomputed_path": str(pre_path.relative_to(root)),
            "samples": st["samples"],
            "features": st["features"],
            "nonzero": st["nonzero"],
            "density": round(st["density"], 6),
        }
        manifest_entries.append(entry)
        print(f"generated: {entry['path']} ({entry['samples']} rows)")
        print(f"generated: {entry['precomputed_path']} ({entry['samples']} rows)")

    manifest = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": "scripts/generate_differential_datasets.py",
        "datasets": manifest_entries,
    }
    manifest_path = ref_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote manifest: {manifest_path.relative_to(root)}")


if __name__ == "__main__":
    main()
