#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Export one differential-suite folder to parity JSON.

The C tool and the Rust tool write the same LIBSVM text.
float() reads each float token from that text.
An empty object marks a file that one side does not have.
A present file that cannot be parsed stops the export.
The export stops when the folder has no model or prediction file.
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NoReturn, TypedDict

# Header floats, sv_coef and plain predictions use %.17g.
# Support-vector feature values use %.8g.
# Probability-prediction columns use %g (six digits).
# svm_type and kernel_type are words.
# Those words must match before the numbers are written.
# degree, nr_class, total_sv, label and nr_sv are integers.
# gamma, coef0, rho, probA, probB and prob_density_marks are floats.

INT_TOKEN = re.compile(r"[+-]?\d+")
PROB_MODEL = re.compile(r"^([cr])_(.+)\.prob\.model$")
PROB_PRED = re.compile(r"^([cr])_(.+)\.prob\.pred$")
PLAIN_MODEL = re.compile(r"^([cr])_(.+)\.model$")
PLAIN_PRED = re.compile(r"^([cr])_(.+)\.pred$")

# A probability model can carry different header values from the plain model.
# LIBSVM prints degree on a polynomial model.
# LIBSVM prints coef0 on a polynomial or sigmoid model.
PLAIN = (
    ("gamma", "gamma"),
    ("coef0", "coef0"),
    ("degree", "degree"),
    ("rho", "rho"),
    ("nr_class", "nr_class"),
    ("total_sv", "total_sv"),
    ("sv_coef", "sv_coef"),
    ("sv", "sv"),
    ("nr_sv", "nr_sv"),
    ("label", "label"),
)
# One-class probability models print prob_density_marks.
# That line is on the probability model, next to probA and probB.
PROB = (
    ("prob_gamma", "gamma"),
    ("prob_coef0", "coef0"),
    ("prob_degree", "degree"),
    ("prob_rho", "rho"),
    ("prob_nr_class", "nr_class"),
    ("prob_total_sv", "total_sv"),
    ("probA", "probA"),
    ("probB", "probB"),
    ("prob_density_marks", "prob_density_marks"),
    ("prob_sv_coef", "sv_coef"),
    ("prob_sv", "sv"),
    ("prob_nr_sv", "nr_sv"),
    ("prob_label", "label"),
)
FIELDS = (
    "gamma",
    "coef0",
    "degree",
    "rho",
    "nr_class",
    "total_sv",
    "sv_coef",
    "sv",
    "predict",
    "predict_probability",
    "nr_sv",
    "label",
    "prob_gamma",
    "prob_coef0",
    "prob_degree",
    "prob_rho",
    "prob_nr_class",
    "prob_total_sv",
    "probA",
    "probB",
    "prob_density_marks",
    "prob_sv_coef",
    "prob_sv",
    "prob_nr_sv",
    "prob_label",
)

Number = float | str
IntList = list[int]
FloatList = list[Number]
FeatureRow = list[int | Number]


class Model(TypedDict):
    svm_type: tuple[str, ...]
    kernel_type: tuple[str, ...]
    gamma: FloatList
    coef0: FloatList
    degree: IntList
    rho: FloatList
    nr_class: IntList
    total_sv: IntList
    probA: FloatList
    probB: FloatList
    prob_density_marks: FloatList
    sv_coef: list[FloatList]
    sv: list[FeatureRow]
    nr_sv: IntList
    label: IntList


CaseValue = Sequence[Any] | dict[str, Any]
ExportDoc = dict[str, dict[str, CaseValue]]
SideMap = dict[str, dict[str, dict[str, Path]]]


class Missing:
    """Marker for a file that is not in the suite folder."""


class ParseError(Exception):
    """Raised when a present file has a bad line."""

    def __init__(self, line: int, detail: str) -> None:
        self.line = line
        self.detail = detail
        super().__init__(detail)


class ProbText:
    """Numeric rows from a probability prediction file, plus any word header."""

    def __init__(self, rows: list[list[int | Number]], words: tuple[str, ...]) -> None:
        self.rows = rows
        self.words = words


def die(msg: str) -> NoReturn:
    print(f"parity_export: error: {msg}", file=sys.stderr)
    raise SystemExit(2)


def stop_parse(path: Path, exc: ParseError) -> NoReturn:
    die(
        f"Cannot parse {path} at line {exc.line}. {exc.detail} "
        "Fix that line and run the export again."
    )


def as_float(token: str) -> Number:
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


def float_token(token: str, line: int, what: str) -> Number:
    try:
        return as_float(token)
    except ValueError:
        raise ParseError(line, f"The {what} token is not a float.") from None


def int_token(token: str, line: int, what: str) -> int:
    try:
        return as_int(token)
    except ValueError:
        raise ParseError(line, f"The {what} token is not an integer.") from None


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


def index_files(suite: Path) -> SideMap:
    found: SideMap = {}
    for path in sorted(suite.iterdir()):
        if not path.is_file():
            continue
        hit = classify(path.name)
        if hit is None:
            continue
        side, role, case = hit
        found.setdefault(case, {}).setdefault(side, {})[role] = path
    return found


def header_numbers(
    header: dict[str, list[str]], header_line: dict[str, int], key: str
) -> FloatList:
    if key not in header:
        return []
    line = header_line[key]
    return [float_token(token, line, key) for token in header[key]]


def header_ints(
    header: dict[str, list[str]], header_line: dict[str, int], key: str
) -> IntList:
    if key not in header:
        return []
    line = header_line[key]
    return [int_token(token, line, key) for token in header[key]]


def parse_model(text: str) -> Model:
    lines = text.splitlines()
    header: dict[str, list[str]] = {}
    header_line: dict[str, int] = {}
    sv_line = 0
    for index, line in enumerate(lines, start=1):
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "SV" and len(parts) == 1:
            sv_line = index
            break
        if parts[0] in header:
            raise ParseError(index, f"Header key {parts[0]} is repeated.")
        header[parts[0]] = parts[1:]
        header_line[parts[0]] = index
    if sv_line == 0:
        raise ParseError(max(len(lines), 1), "The model has no SV line.")
    for key in ("svm_type", "kernel_type", "nr_class", "rho"):
        if key not in header:
            raise ParseError(sv_line, f"The model has no {key} line.")
    nr_line = header_line["nr_class"]
    if len(header["nr_class"]) != 1:
        raise ParseError(nr_line, "The nr_class line needs one integer.")
    nr_class = int_token(header["nr_class"][0], nr_line, "nr_class")
    # Each support vector starts with nr_class-1 coefficients.
    ncoef = nr_class - 1
    if ncoef < 0:
        raise ParseError(nr_line, "nr_class must be 1 or more.")
    sv_coef: list[FloatList] = []
    sv_rows: list[FeatureRow] = []
    for index, raw in enumerate(lines[sv_line:], start=sv_line + 1):
        parts = raw.split()
        if not parts:
            continue
        if len(parts) < ncoef:
            raise ParseError(index, "The support-vector row has too few coefficients.")
        coefs = [float_token(token, index, "sv_coef") for token in parts[:ncoef]]
        features: FeatureRow = []
        for token in parts[ncoef:]:
            if ":" not in token:
                raise ParseError(index, "The token has no feature index.")
            idx_text, val_text = token.split(":", 1)
            features.append(int_token(idx_text, index, "feature index"))
            features.append(float_token(val_text, index, "feature value"))
        sv_coef.append(coefs)
        sv_rows.append(features)
    return {
        "svm_type": tuple(header["svm_type"]),
        "kernel_type": tuple(header["kernel_type"]),
        "gamma": header_numbers(header, header_line, "gamma"),
        "coef0": header_numbers(header, header_line, "coef0"),
        "degree": header_ints(header, header_line, "degree"),
        "rho": header_numbers(header, header_line, "rho"),
        "nr_class": header_ints(header, header_line, "nr_class"),
        "total_sv": header_ints(header, header_line, "total_sv"),
        "probA": header_numbers(header, header_line, "probA"),
        "probB": header_numbers(header, header_line, "probB"),
        "prob_density_marks": header_numbers(header, header_line, "prob_density_marks"),
        "sv_coef": sv_coef,
        "sv": sv_rows,
        "nr_sv": header_ints(header, header_line, "nr_sv"),
        "label": header_ints(header, header_line, "label"),
    }


def parse_scalar(text: str) -> FloatList:
    values: FloatList = []
    for index, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 1:
            raise ParseError(index, "The prediction line needs one value.")
        values.append(float_token(parts[0], index, "prediction"))
    return values


def parse_prob(text: str) -> ProbText:
    # One-class probability files name the columns with words.
    # The JSON stores the numeric columns only.
    # The exporter compares the words first.
    lines = [line.split() for line in text.splitlines() if line.split()]
    raw_lines = [index for index, line in enumerate(text.splitlines(), start=1) if line.split()]
    rows: list[list[int | Number]] = []
    words: tuple[str, ...] = ()
    start = 0
    if lines and lines[0][0] == "labels":
        rest = lines[0][1:]
        if all(INT_TOKEN.fullmatch(token) for token in rest):
            rows.append([int(token) for token in rest])
        else:
            words = tuple(rest)
        start = 1
    for parts, index in zip(lines[start:], raw_lines[start:]):
        rows.append([float_token(token, index, "probability") for token in parts])
    return ProbText(rows, words)


def read_text(path: Path) -> str:
    try:
        data = path.read_bytes()
    except OSError as exc:
        detail = exc.strerror or "the read failed"
        die(
            f"Cannot read {path} at line 1. {detail}. "
            "Check the permissions and run the export again."
        )
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        line = data[: exc.start].count(b"\n") + 1
        die(
            f"Cannot read {path} at line {line}. The file is not UTF-8. "
            "Save the file as UTF-8 and run the export again."
        )


def read_model(path: Path | None) -> Model | Missing:
    if path is None:
        return Missing()
    try:
        return parse_model(read_text(path))
    except ParseError as exc:
        stop_parse(path, exc)


def read_pred(path: Path | None, scalar: bool) -> FloatList | ProbText | Missing:
    if path is None:
        return Missing()
    try:
        text = read_text(path)
        if scalar:
            return parse_scalar(text)
        return parse_prob(text)
    except ParseError as exc:
        stop_parse(path, exc)


def take_model(model: Model | Missing, key: str) -> CaseValue:
    # An empty object is not a numeric array.
    # A missing file then has a different shape from a present file.
    if not isinstance(model, dict):
        return {}
    return model[key]


def put_model(
    c_doc: ExportDoc,
    r_doc: ExportDoc,
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


def body(value: FloatList | ProbText | Missing) -> CaseValue:
    if isinstance(value, ProbText):
        return value.rows
    if isinstance(value, list):
        return value
    return {}


def put_pred(
    c_doc: ExportDoc,
    r_doc: ExportDoc,
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


def export(suite: Path) -> tuple[ExportDoc, ExportDoc]:
    c_doc: ExportDoc = {key: {} for key in FIELDS}
    r_doc: ExportDoc = {key: {} for key in FIELDS}
    files = index_files(suite)
    if not files:
        die(
            "The suite folder holds no model or prediction file. "
            "Add a model or a prediction file and run the export again."
        )
    for case in sorted(files):
        sides = files[case]
        put_model(c_doc, r_doc, case, sides, "model", PLAIN)
        put_model(c_doc, r_doc, case, sides, "prob_model", PROB)
        put_pred(c_doc, r_doc, case, sides, "pred", "predict", True)
        put_pred(c_doc, r_doc, case, sides, "prob_pred", "predict_probability", False)
    return c_doc, r_doc


def write_json(path: Path, doc: ExportDoc) -> None:
    path.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv: list[str]) -> int:
    # argv is sys.argv. The suite folder is the only argument.
    root = Path(__file__).resolve().parent.parent
    suite = root / ".tmp" / "differential_suite"
    if len(argv) == 2:
        suite = Path(argv[1])
    elif len(argv) != 1:
        die("Pass the suite folder only. Run the export with one folder path.")
    if not suite.is_dir():
        die(f"The suite folder {suite} does not exist. Create it and run the export again.")
    c_doc, r_doc = export(suite)
    if list(c_doc) != list(r_doc):
        die("The two JSON files have different keys. Run the export again.")
    for key in c_doc:
        if list(c_doc[key]) != list(r_doc[key]):
            die(f"Field {key} has different case keys on the two sides. Run the export again.")
    c_path = suite / "parity_c.json"
    r_path = suite / "parity_r.json"
    write_json(c_path, c_doc)
    write_json(r_path, r_doc)
    print("Header floats and sv_coef use %.17g.")
    print("degree, nr_class, total_sv, nr_sv and label are integers.")
    print("Support-vector feature values use %.8g.")
    print("Plain predictions use %.17g.")
    print("Probability-prediction columns use %g.")
    print(c_path)
    print(r_path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
