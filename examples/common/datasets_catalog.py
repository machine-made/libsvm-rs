#!/usr/bin/env python3
"""Dataset catalog for scientific demos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

BASE = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets"


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    title: str
    task: str
    svm_type: int
    features: int
    train_url: Optional[str] = None
    test_url: Optional[str] = None
    single_url: Optional[str] = None
    train_rows: Optional[int] = None
    test_rows: Optional[int] = None

    @property
    def needs_split(self) -> bool:
        return self.single_url is not None


DATASETS: Dict[str, DatasetSpec] = {
    "ijcnn1": DatasetSpec(
        dataset_id="ijcnn1",
        title="IJCNN 2001 (binary classification)",
        task="classification",
        svm_type=0,
        features=22,
        train_url=f"{BASE}/binary/ijcnn1.bz2",
        test_url=f"{BASE}/binary/ijcnn1.t.bz2",
        train_rows=49_990,
        test_rows=91_701,
    ),
    "covtype_scale": DatasetSpec(
        dataset_id="covtype_scale",
        title="Covertype scale (multiclass classification)",
        task="classification",
        svm_type=0,
        features=54,
        single_url=f"{BASE}/multiclass/covtype.scale.bz2",
        train_rows=464_810,
        test_rows=116_202,
    ),
    "yearpredictionmsd": DatasetSpec(
        dataset_id="yearpredictionmsd",
        title="YearPredictionMSD (regression)",
        task="regression",
        svm_type=3,
        features=90,
        train_url=f"{BASE}/regression/YearPredictionMSD.bz2",
        test_url=f"{BASE}/regression/YearPredictionMSD.t.bz2",
        train_rows=463_715,
        test_rows=51_630,
    ),
    "higgs": DatasetSpec(
        dataset_id="higgs",
        title="HIGGS (binary classification)",
        task="classification",
        svm_type=0,
        features=28,
        single_url=f"{BASE}/binary/HIGGS.xz",
        train_rows=10_500_000,
        test_rows=500_000,
    ),
    "rcv1_binary": DatasetSpec(
        dataset_id="rcv1_binary",
        title="RCV1 binary (sparse text classification)",
        task="classification",
        svm_type=0,
        features=47_236,
        train_url=f"{BASE}/binary/rcv1_train.binary.bz2",
        test_url=f"{BASE}/binary/rcv1_test.binary.bz2",
        train_rows=20_242,
        test_rows=677_399,
    ),
}


def list_option_dataset_ids() -> Dict[str, List[str]]:
    return {
        "option1": ["ijcnn1"],
        "option2": ["ijcnn1", "covtype_scale", "yearpredictionmsd"],
        "option3": ["ijcnn1", "covtype_scale", "yearpredictionmsd", "higgs"],
        "option4": ["ijcnn1", "covtype_scale", "yearpredictionmsd", "rcv1_binary"],
        "option5": [
            "ijcnn1",
            "covtype_scale",
            "yearpredictionmsd",
            "higgs",
            "rcv1_binary",
        ],
    }
