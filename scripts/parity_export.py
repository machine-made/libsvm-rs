#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Export one differential-suite folder to parity JSON.

The C tool and the Rust tool write the same LIBSVM text.
float() reads each numeric token from that text.
An empty object marks a file that one side does not have.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

# Header floats, sv_coef and plain predictions use %.17g.
# Support-vector feature values use %.8g.
# Probability-prediction columns use %g (six digits).

INT_TOKEN = re.compile(r"[+-]?\d+")
PROB_MODEL = re.compile(r"^([cr])_(.+)\.prob\.model$")
PROB_PRED = re.compile(r"^([cr])_(.+)\.prob\.pred$")
PLAIN_MODEL = re.compile(r"^([cr])_(.+)\.model$")
PLAIN_PRED = re.compile(r"^([cr])_(.+)\.pred$")

# Plain-model fields, then the same kinds from the probability model.
# A missing probability file must not change the shape of the plain fields.
PLAIN = (
    ("gamma", "gamma"),
    ("rho", "rho"),
    ("sv_coef", "sv_coef"),
    ("sv", "sv"),
    ("nr_sv", "nr_sv"),
    ("label", "label"),
)
PROB = (
    ("prob_gamma", "gamma"),
    ("prob_rho", "rho"),
    ("probA", "probA"),
    ("probB", "probB"),
    ("prob_sv_coef", "sv_coef"),
    ("prob_sv", "sv"),
    ("prob_nr_sv", "nr_sv"),
    ("prob_label", "label"),
)
FIELDS = (
    "gamma",
    "rho",
    "probA",
    "probB",
    "sv_coef",
    "sv",
    "predict",
    "predict_probability",
    "nr_sv",
    "label",
    "prob_gamma",
    "prob_rho",
    "prob_sv_coef",
    "prob_sv",
    "prob_nr_sv",
    "prob_label",
)


class Missing:
    """Marker for a missing file or a file that float() cannot read."""


class ProbText:
    """Numeric rows from a probability prediction file, plus any word header."""

    def __init__(self, rows: list, words: tuple[str, ...]) -> None:
        self.rows = rows
        self.words = words


def die(msg: str) -> None:
    print(f"parity_export: error: {msg}", file=sys.stderr)
    raise SystemExit(2)


def as_float(token: str) -> float | str:
    value = float(token)
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    return value


def as_int(token: str) -> int:
    if INT_TOKEN.fullmatch(token) is None:
        raise ValueError(token)
    return int(token)


def classify(name: str) -> tuple[str, str, str] | None:
    # Match the probability names first. They also end in .model and .pred.
    for role, pattern in (
        ("prob_model", PROB_MODEL),
        ("prob_pred", PROB_PRED),
        ("model", PLAIN_MODEL),
        ("pred", PLAIN_PRED),
    ):
        match = pattern.fullmatch(name)
        if match is not None:
            return match.group(1), role, match.group(2)
    return None


def index_files(suite: Path) -> dict[str, dict[str, dict[str, Path]]]:
    found: dict[str, dict[str, dict[str, Path]]] = {}
    for path in sorted(suite.iterdir()):
        if not path.is_file():
            continue
        hit = classify(path.name)
        if hit is None:
            continue
        side, role, case = hit
        found.setdefault(case, {}).setdefault(side, {})[role] = path
    return found


def header_numbers(header: dict[str, list[str]], key: str) -> list:
    if key not in header:
        return []
    return [as_float(token) for token in header[key]]


def header_ints(header: dict[str, list[str]], key: str) -> list[int]:
    if key not in header:
        return []
    return [as_int(token) for token in header[key]]


def parse_model(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8").splitlines()
    header: dict[str, list[str]] = {}
    sv_at = None
    for index, line in enumerate(lines):
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "SV" and len(parts) == 1:
            sv_at = index
            break
        if parts[0] in header:
            raise ValueError(parts[0])
        header[parts[0]] = parts[1:]
    if sv_at is None:
        raise ValueError("SV")
    for key in ("svm_type", "kernel_type", "nr_class", "rho"):
        if key not in header:
            raise ValueError(key)
    if len(header["nr_class"]) != 1:
        raise ValueError("nr_class")
    nr_class = as_int(header["nr_class"][0])
    # Each support vector starts with nr_class-1 coefficients.
    ncoef = nr_class - 1
    if ncoef < 0:
        raise ValueError("nr_class")
    sv_coef: list[list] = []
    sv_rows: list[list] = []
    for raw in lines[sv_at + 1 :]:
        parts = raw.split()
        if not parts:
            continue
        if len(parts) < ncoef:
            raise ValueError("sv_coef")
        coefs = [as_float(token) for token in parts[:ncoef]]
        features: list = []
        for token in parts[ncoef:]:
            if ":" not in token:
                raise ValueError(token)
            idx_text, val_text = token.split(":", 1)
            features.append(as_int(idx_text))
            features.append(as_float(val_text))
        sv_coef.append(coefs)
        sv_rows.append(features)
    return {
        "svm_type": tuple(header["svm_type"]),
        "kernel_type": tuple(header["kernel_type"]),
        "gamma": header_numbers(header, "gamma"),
        "rho": header_numbers(header, "rho"),
        "probA": header_numbers(header, "probA"),
        "probB": header_numbers(header, "probB"),
        "sv_coef": sv_coef,
        "sv": sv_rows,
        "nr_sv": header_ints(header, "nr_sv"),
        "label": header_ints(header, "label"),
    }


def parse_scalar(text: str) -> list:
    values = []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 1:
            raise ValueError(line)
        values.append(as_float(parts[0]))
    return values


def parse_prob(text: str) -> ProbText:
    # One-class probability files name the columns with words.
    # The JSON stores the numeric columns only.
    # The exporter compares the words first.
    lines = [line.split() for line in text.splitlines() if line.split()]
    rows: list = []
    words: tuple[str, ...] = ()
    start = 0
    if lines and lines[0][0] == "labels":
        rest = lines[0][1:]
        if all(INT_TOKEN.fullmatch(token) for token in rest):
            rows.append([int(token) for token in rest])
        else:
            words = tuple(rest)
        start = 1
    for parts in lines[start:]:
        rows.append([as_float(token) for token in parts])
    return ProbText(rows, words)


def read_model(path: Path | None) -> dict | Missing:
    if path is None:
        return Missing()
    try:
        return parse_model(path)
    except (OSError, UnicodeError, ValueError):
        return Missing()


def read_pred(path: Path | None, scalar: bool) -> list | ProbText | Missing:
    if path is None:
        return Missing()
    try:
        text = path.read_text(encoding="utf-8")
        if scalar:
            return parse_scalar(text)
        return parse_prob(text)
    except (OSError, UnicodeError, ValueError):
        return Missing()


def take_model(model: dict | Missing, key: str):
    # An empty object is not a numeric array.
    # A missing file then has a different shape from a present file.
    if not isinstance(model, dict):
        return {}
    return model[key]


def put_model(
    c_doc: dict,
    r_doc: dict,
    case: str,
    sides: dict[str, dict[str, Path]],
    role: str,
    mapping: tuple[tuple[str, str], ...],
) -> None:
    c_path = sides.get("c", {}).get(role)
    r_path = sides.get("r", {}).get(role)
    if c_path is None and r_path is None:
        return
    c_model = read_model(c_path)
    r_model = read_model(r_path)
    # A different svm_type or kernel_type is not the same model.
    # The Rust side then uses an empty object.
    if isinstance(c_model, dict) and isinstance(r_model, dict):
        c_id = (c_model["svm_type"], c_model["kernel_type"])
        r_id = (r_model["svm_type"], r_model["kernel_type"])
        if c_id != r_id:
            r_model = Missing()
    for out_name, src_name in mapping:
        c_doc[out_name][case] = take_model(c_model, src_name)
        r_doc[out_name][case] = take_model(r_model, src_name)


def body(value: list | ProbText | Missing):
    if isinstance(value, ProbText):
        return value.rows
    if isinstance(value, list):
        return value
    return {}


def put_pred(
    c_doc: dict,
    r_doc: dict,
    case: str,
    sides: dict[str, dict[str, Path]],
    role: str,
    field: str,
    scalar: bool,
) -> None:
    c_path = sides.get("c", {}).get(role)
    r_path = sides.get("r", {}).get(role)
    if c_path is None and r_path is None:
        return
    c_value = read_pred(c_path, scalar)
    r_value = read_pred(r_path, scalar)
    if isinstance(c_value, ProbText) and isinstance(r_value, ProbText):
        if c_value.words != r_value.words:
            r_value = Missing()
    c_doc[field][case] = body(c_value)
    r_doc[field][case] = body(r_value)


def export(suite: Path) -> tuple[dict, dict]:
    c_doc = {key: {} for key in FIELDS}
    r_doc = {key: {} for key in FIELDS}
    files = index_files(suite)
    for case in sorted(files):
        sides = files[case]
        put_model(c_doc, r_doc, case, sides, "model", PLAIN)
        put_model(c_doc, r_doc, case, sides, "prob_model", PROB)
        put_pred(c_doc, r_doc, case, sides, "pred", "predict", True)
        put_pred(c_doc, r_doc, case, sides, "prob_pred", "predict_probability", False)
    return c_doc, r_doc


def write_json(path: Path, doc: dict) -> None:
    path.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv: list[str]) -> int:
    # argv is sys.argv. The suite folder is the only argument.
    root = Path(__file__).resolve().parent.parent
    suite = root / ".tmp" / "differential_suite"
    if len(argv) == 2:
        suite = Path(argv[1])
    elif len(argv) != 1:
        die("Pass the suite folder only.")
    if not suite.is_dir():
        die(f"The suite folder {suite} does not exist.")
    c_doc, r_doc = export(suite)
    if list(c_doc) != list(r_doc):
        die("The two JSON files have different keys.")
    for key in c_doc:
        if list(c_doc[key]) != list(r_doc[key]):
            die(f"Field {key} has different case keys on the two sides.")
    c_path = suite / "parity_c.json"
    r_path = suite / "parity_r.json"
    write_json(c_path, c_doc)
    write_json(r_path, r_doc)
    print("Header floats and sv_coef use %.17g.")
    print("Support-vector feature values use %.8g.")
    print("Plain predictions use %.17g.")
    print("Probability-prediction columns use %g.")
    print(c_path)
    print(r_path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
