#!/usr/bin/env python3
"""Download and verify reference datasets used by the scientific demos."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

from datasets_catalog import DATASETS, list_option_dataset_ids


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
    )
    if result.returncode != 0:
        raise RuntimeError(f"curl failed ({result.returncode}) for {url}")
    os.replace(tmp, dest)


def files_for_dataset(dataset_id: str) -> List[tuple[str, str]]:
    spec = DATASETS[dataset_id]
    out: List[tuple[str, str]] = []
    if spec.train_url:
        out.append((spec.train_url, Path(spec.train_url).name))
    if spec.test_url:
        out.append((spec.test_url, Path(spec.test_url).name))
    if spec.single_url:
        out.append((spec.single_url, Path(spec.single_url).name))
    return out


def expand_dataset_args(values: Iterable[str]) -> List[str]:
    options = list_option_dataset_ids()
    chosen: List[str] = []
    for v in values:
        if v == "all":
            for ds in DATASETS:
                if ds not in chosen:
                    chosen.append(ds)
            continue
        if v in options:
            for ds in options[v]:
                if ds not in chosen:
                    chosen.append(ds)
            continue
        if v not in DATASETS:
            valid = sorted(list(DATASETS.keys()) + list(options.keys()) + ["all"])
            raise SystemExit(f"Unknown dataset/option '{v}'. Valid values: {', '.join(valid)}")
        if v not in chosen:
            chosen.append(v)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help="Dataset IDs, option names (option1..option5), or 'all'",
    )
    parser.add_argument(
        "--raw-dir",
        default="examples/data/raw",
        help="Destination directory for raw downloaded artifacts",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if file exists",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    raw_dir = (root / args.raw_dir).resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)

    dataset_ids = expand_dataset_args(args.datasets)
    print(f"Datasets selected: {', '.join(dataset_ids)}")

    manifest: Dict[str, object] = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "raw_dir": str(raw_dir),
        "datasets": {},
    }

    for dataset_id in dataset_ids:
        spec = DATASETS[dataset_id]
        print(f"\n[{dataset_id}] {spec.title}")
        entries: List[Dict[str, object]] = []
        for url, name in files_for_dataset(dataset_id):
            dest = raw_dir / name
            if dest.exists() and not args.force:
                print(f"- Reusing {name}")
            else:
                print(f"- Downloading {name}")
                download(url, dest)
            file_sha = sha256(dest)
            size = dest.stat().st_size
            print(f"  size={size} bytes sha256={file_sha[:16]}...")
            entries.append(
                {
                    "url": url,
                    "file": str(dest.relative_to(root)),
                    "size_bytes": size,
                    "sha256": file_sha,
                }
            )
        manifest["datasets"][dataset_id] = {
            "title": spec.title,
            "task": spec.task,
            "files": entries,
            "expected_train_rows": spec.train_rows,
            "expected_test_rows": spec.test_rows,
        }

    out = root / "examples" / "data" / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote manifest: {out}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        raise SystemExit(130)
