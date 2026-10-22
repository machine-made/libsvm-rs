#!/usr/bin/env python3
"""
Generate LIBSVM precomputed-kernel datasets from sparse LIBSVM input files.

Outputs files next to inputs with suffix ".precomputed".
Kernel used: linear dot product K(x_i, x_j) = x_i · x_j.

Usage:
  python3 scripts/generate_precomputed_datasets.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple


def parse_libsvm(path: Path) -> Tuple[List[float], List[Dict[int, float]]]:
    labels: List[float] = []
    rows: List[Dict[int, float]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            try:
                y = float(parts[0])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: invalid label") from exc

            feats: Dict[int, float] = {}
            for tok in parts[1:]:
                if ":" not in tok:
                    raise ValueError(f"{path}:{line_no}: invalid token '{tok}'")
                idx_s, val_s = tok.split(":", 1)
                try:
                    idx = int(idx_s)
                    val = float(val_s)
                except ValueError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid token '{tok}'") from exc
                feats[idx] = val

            labels.append(y)
            rows.append(feats)

    return labels, rows


def sparse_dot(a: Dict[int, float], b: Dict[int, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    total = 0.0
    for k, va in a.items():
        vb = b.get(k)
        if vb is not None:
            total += va * vb
    return total


def write_precomputed(path: Path, labels: List[float], rows: List[Dict[int, float]]) -> None:
    n = len(rows)
    out_path = path.with_suffix(path.suffix + ".precomputed")

    # Precompute full symmetric kernel matrix.
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


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"

    datasets = [
        data_dir / "heart_scale",
        data_dir / "iris.scale",
        data_dir / "housing_scale",
    ]

    for ds in datasets:
        labels, rows = parse_libsvm(ds)
        write_precomputed(ds, labels, rows)
        print(f"generated: {ds.name}.precomputed ({len(rows)} rows)")


if __name__ == "__main__":
    main()
