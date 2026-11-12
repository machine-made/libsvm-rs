#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Compare two export files and write the ULP report.

Each float is compared at f64 or f32.
The ULP count uses the IEEE-754 bit pattern in float order.
Exit 0 when every field is inside the tolerance.
Exit 1 when one field is not, or when the files hold no number.
Exit 2 when an input is not usable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn

VERSION = 1
FAILURE_CAP = 10
TOLERANCE_KEYS = {"rtol", "atol", "ulp", "reason", "precision", "signed_zero"}
SPECIAL = {
    "nan": math.nan,
    "infinity": math.inf,
    "-infinity": -math.inf,
    "inf": math.inf,
    "-inf": -math.inf,
}
Leaf = int | float | bool


def reject(message: str) -> NoReturn:
    print(f"parity_compare: error: {message}", file=sys.stderr)
    raise SystemExit(2)


def stored_path(path: Path) -> str:
    # The report is compared on another machine, so the paths stay relative.
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_document(path: Path, role: str) -> Any:
    if not path.is_file():
        reject(
            f"The {role} file {path} does not exist. "
            "Pass the export JSON and run the compare again."
        )
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        detail = exc.strerror or "the read failed"
        reject(
            f"Cannot read the {role} file {path}. {detail}. "
            "Check the path and run the compare again."
        )
    except UnicodeError:
        reject(
            f"The {role} file {path} is not UTF-8. "
            "Save the file as UTF-8 and run the compare again."
        )

    def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        obj: dict[str, Any] = {}
        for key, value in pairs:
            if key in obj:
                reject(
                    f"The {role} file {path} repeats the key {key!r}. "
                    "Remove the repeated key and run the compare again."
                )
            obj[key] = value
        return obj

    try:
        return json.loads(text, object_pairs_hook=unique_keys)
    except json.JSONDecodeError as exc:
        reject(
            f"The {role} file {path} is not valid JSON "
            f"({exc.msg} at line {exc.lineno}). "
            "Write the file again and run the compare again."
        )


def reject_unless_f64(value: int | float, where: str, role: str) -> None:
    """Reject a number that does not fit an f64."""
    try:
        number = float(value)
    except OverflowError:
        number = math.nan
    if not math.isfinite(number):
        reject(
            f"The {role} value at {where} does not fit an f64. "
            "Use a finite number and run the compare again."
        )


def as_leaf(value: object, where: str, role: str) -> Leaf:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        # A JSON number has no Infinity token. Overflow must not become infinity.
        reject_unless_f64(value, where, role)
        return value
    if isinstance(value, str):
        key = value.strip().lower()
        if key in SPECIAL:
            return SPECIAL[key]
        text = value.strip()
        if text.lower().startswith(("0x", "-0x", "+0x")):
            try:
                number = float.fromhex(text)
            except ValueError:
                number = None
            if number is not None:
                reject_unless_f64(number, where, role)
                return number
        reject(
            f"The {role} value at {where} is the string {value!r}, not a number. "
            "Use a JSON number, NaN, Infinity or -Infinity."
        )
    reject(
        f"The {role} value at {where} is a {type(value).__name__}, not a number. "
        "Use a JSON number and run the compare again."
    )


def field_token(key: str) -> str:
    # A dot in a key must not look like the separator of a nested key.
    return key.replace("\\", "\\\\").replace(".", "\\.")


def field_path(parent: str, key: str) -> str:
    token = field_token(key)
    if parent == "":
        return token
    return f"{parent}.{token}"


@dataclass
class Series:
    nodes: list[tuple[str, str]] = field(default_factory=list)
    values: list[tuple[str, Leaf]] = field(default_factory=list)


def collect_fields(document: object, role: str) -> dict[str, Series]:
    found: dict[str, Series] = {}

    def add_field(name: str) -> Series:
        if name in found:
            reject(
                f"The {role} file uses the field path {name!r} twice. "
                "Rename one key so each path is unique."
            )
        series = Series()
        found[name] = series
        return series

    def consume(node: object, series: Series, field_name: str, place: str) -> None:
        if isinstance(node, list):
            series.nodes.append((place, f"an array of {len(node)}"))
            for index, item in enumerate(node):
                consume(item, series, field_name, f"{place}[{index}]")
            return
        if isinstance(node, dict):
            reject(
                f"The {role} file has an object in an array at {field_name}{place}. "
                "Put only numbers in an array and run the compare again."
            )
        series.nodes.append((place, "a number"))
        series.values.append((place, as_leaf(node, f"{field_name}{place}", role)))

    def descend(node: object, path: str) -> None:
        if isinstance(node, dict):
            if len(node) == 0:
                add_field(path).nodes.append(("", "an empty object"))
                return
            for key, item in node.items():
                descend(item, field_path(path, key))
            return
        consume(node, add_field(path), path, "")

    if isinstance(document, dict):
        for key, item in document.items():
            descend(item, field_path("", key))
        return found
    consume(document, add_field("$"), "$", "")
    return found


def shape_gap(name: str, left: list[tuple[str, str]], right: list[tuple[str, str]]) -> str | None:
    width = max(len(left), len(right))
    for index in range(width):
        a = left[index] if index < len(left) else None
        b = right[index] if index < len(right) else None
        if a == b:
            continue
        place = (a or b or ("", ""))[0]
        left_kind = a[1] if a else "nothing"
        right_kind = b[1] if b else "nothing"
        return (
            f"shape differs at {name}{place}: "
            f"the reference has {left_kind}, the candidate has {right_kind}"
        )
    return None


@dataclass(frozen=True)
class Tolerance:
    rtol: float = 0.0
    atol: float = 0.0
    ulp: int | None = None
    reason: str = ""
    precision: str = "f64"
    signed_zero: str = "strict"

    def as_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "rtol": self.rtol,
            "atol": self.atol,
            "precision": self.precision,
            "signed_zero": self.signed_zero,
            "reason": self.reason,
        }
        if self.ulp is not None:
            out["ulp"] = self.ulp
        return out

    def width(self) -> int:
        return 32 if self.precision == "f32" else 64


def parse_tolerance(raw: object, where: str) -> Tolerance:
    if not isinstance(raw, dict):
        reject(
            f"The tolerance {where} must be an object. "
            'Use {"rtol": 0, "atol": 0, "reason": "..."}.'
        )
    unknown = sorted(set(raw) - TOLERANCE_KEYS)
    if unknown:
        reject(
            f"The tolerance {where} has unknown keys {unknown}. "
            f"Use only {sorted(TOLERANCE_KEYS)}."
        )
    reason = raw.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        reject(
            f"The tolerance {where} has no reason. "
            "Add a reason and run the compare again."
        )
    rtol = 0.0
    atol = 0.0
    ulp: int | None = None
    precision = "f64"
    signed_zero = "strict"
    for key in ("rtol", "atol"):
        if key not in raw:
            continue
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, int | float):
            reject(f"The tolerance {where}.{key} must be a finite number, zero or more.")
        try:
            number = float(value)
        except OverflowError:
            number = math.nan
        if not math.isfinite(number):
            reject(
                f"The tolerance {where}.{key} does not fit an f64. "
                "Use a finite number and run the compare again."
            )
        if number < 0:
            reject(f"The tolerance {where}.{key} must be a finite number, zero or more.")
        if key == "rtol":
            rtol = number
        else:
            atol = number
    if "ulp" in raw:
        value = raw["ulp"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            reject(f"The tolerance {where}.ulp must be a whole number, zero or more.")
        ulp = value
    if "precision" in raw:
        precision = raw["precision"]
        if precision not in ("f64", "f32"):
            reject(f'The tolerance {where}.precision must be "f64" or "f32".')
    if "signed_zero" in raw:
        signed_zero = raw["signed_zero"]
        if signed_zero not in ("strict", "ignore"):
            reject(f'The tolerance {where}.signed_zero must be "strict" or "ignore".')
    return Tolerance(rtol, atol, ulp, reason.strip(), precision, signed_zero)


def parse_config(document: object) -> tuple[Tolerance | None, dict[str, Tolerance]]:
    if not isinstance(document, dict):
        reject(
            'The tolerance file must be an object with "default" or "fields". '
            "Fix the file and run the compare again."
        )
    unknown = sorted(set(document) - {"default", "fields", "comment"})
    if unknown:
        reject(
            f"The tolerance file has unknown keys {unknown}. "
            "Use default, fields or comment."
        )
    default = parse_tolerance(document["default"], "default") if "default" in document else None
    raw_fields = document.get("fields", {})
    if not isinstance(raw_fields, dict):
        reject('The tolerance "fields" value must be an object. Fix it and run the compare again.')
    named = {name: parse_tolerance(raw, f"fields.{name}") for name, raw in raw_fields.items()}
    if default is None and not named:
        reject(
            "The tolerance file has no default and no fields. "
            "Add a default tolerance and run the compare again."
        )
    return default, named


def pack_code(width: int) -> str:
    return "<f" if width == 32 else "<d"


def ordered_rank(value: float, width: int) -> int:
    # A negative float uses the negative magnitude.
    # The two zeros share one rank.
    # One step on this line is one ULP.
    raw = int.from_bytes(struct.pack(pack_code(width), value), "little")
    magnitude = raw & ((1 << (width - 1)) - 1)
    if raw >> (width - 1):
        return -magnitude
    return magnitude


def ulp_steps(left: float, right: float, width: int) -> int:
    return abs(ordered_rank(left, width) - ordered_rank(right, width))


def narrow(value: float, width: int) -> float:
    if width == 64 or not math.isfinite(value):
        return value
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def same_bits(left: float, right: float, width: int) -> bool:
    code = pack_code(width)
    return struct.pack(code, left) == struct.pack(code, right)


def min_normal(width: int) -> float:
    if width == 32:
        return struct.unpack("<f", struct.pack("<I", 0x00800000))[0]
    return sys.float_info.min


def is_subnormal(value: float, width: int) -> bool:
    return value != 0.0 and math.isfinite(value) and abs(value) < min_normal(width)


def exact_pair(left: Leaf, right: Leaf) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return True
    return isinstance(left, int) and isinstance(right, int)


@dataclass
class FieldReport:
    name: str
    tolerance: Tolerance | None
    count: int = 0
    failures: int = 0
    max_abs_err: float = 0.0
    max_rel_err: float = 0.0
    max_ulp: int = 0
    worst_abs_index: str | None = None
    worst_rel_index: str | None = None
    nonfinite_mismatch: int = 0
    sign_mismatch: int = 0
    subnormal_in_ref: int = 0
    subnormal_flushed: int = 0
    int_mismatch: int = 0
    structure: str | None = None
    first_failures: list[dict[str, object]] = field(default_factory=list)

    def passed(self) -> bool:
        return self.structure is None and self.failures == 0

    def note(
        self, place: str, ref: object, cand: object, why: str, extra: dict[str, object] | None = None
    ) -> None:
        self.failures += 1
        if len(self.first_failures) >= FAILURE_CAP:
            return
        row: dict[str, object] = {"index": place, "ref": ref, "cand": cand, "why": why}
        if extra:
            row.update(extra)
        self.first_failures.append(row)

    def as_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "name": self.name,
            "pass": self.passed(),
            "count": self.count,
            "failures": self.failures,
            "tolerance": None if self.tolerance is None else self.tolerance.as_dict(),
            "max_abs_err": self.max_abs_err,
            "max_rel_err": self.max_rel_err,
            "max_ulp": self.max_ulp,
            "worst_abs_index": self.worst_abs_index,
            "worst_rel_index": self.worst_rel_index,
            "nonfinite_mismatch": self.nonfinite_mismatch,
            "sign_mismatch": self.sign_mismatch,
            "subnormal_in_ref": self.subnormal_in_ref,
            "subnormal_flushed": self.subnormal_flushed,
            "int_mismatch": self.int_mismatch,
            "first_failures": self.first_failures,
        }
        if self.structure is not None:
            out["structure"] = self.structure
        return out


def mark_nonfinite(report: FieldReport, place: str, ref: float, cand: float) -> None:
    report.nonfinite_mismatch += 1
    if report.max_abs_err != math.inf:
        report.worst_abs_index = place
    if report.max_rel_err != math.inf:
        report.worst_rel_index = place
    report.max_abs_err = math.inf
    report.max_rel_err = math.inf
    report.note(place, ref, cand, "non-finite values differ")


def compare_float(report: FieldReport, tol: Tolerance, place: str, ref: float, cand: float) -> None:
    width = tol.width()
    try:
        ref_in = float(ref)
        cand_in = float(cand)
    except OverflowError:
        reject(
            f"The value at {report.name}{place} does not fit an f64. "
            "Use a finite number and run the compare again."
        )
    # Finiteness uses the parsed values, before the narrower format.
    # A finite value that overflows that format is not an infinite input.
    ref_finite = math.isfinite(ref_in)
    cand_finite = math.isfinite(cand_in)
    ref_n = narrow(ref_in, width)
    cand_n = narrow(cand_in, width)
    ref_overflow = ref_finite and not math.isfinite(ref_n)
    cand_overflow = cand_finite and not math.isfinite(cand_n)
    if (ref_overflow and not cand_finite) or (cand_overflow and not ref_finite):
        mark_nonfinite(report, place, ref_in, cand_in)
        return
    ref = ref_n
    cand = cand_n
    if is_subnormal(ref, width):
        report.subnormal_in_ref += 1
    if same_bits(ref, cand, width) or (math.isnan(ref) and math.isnan(cand)):
        return
    # No tolerance covers a finite value against NaN or infinity.
    if not math.isfinite(ref) or not math.isfinite(cand):
        mark_nonfinite(report, place, ref, cand)
        return
    if ref == 0.0 and cand == 0.0:
        if tol.signed_zero == "strict":
            report.sign_mismatch += 1
            report.note(place, ref, cand, "signed zero differs")
        return
    if math.copysign(1.0, ref) != math.copysign(1.0, cand):
        report.sign_mismatch += 1
    abs_err = abs(cand - ref)
    rel_err = abs_err / abs(ref) if ref != 0.0 else math.inf
    steps = ulp_steps(ref, cand, width)
    if abs_err > report.max_abs_err:
        report.max_abs_err = abs_err
        report.worst_abs_index = place
    # A zero reference has no finite ratio. Store Infinity so the max stays honest.
    if rel_err > report.max_rel_err:
        report.max_rel_err = rel_err
        report.worst_rel_index = place
    report.max_ulp = max(report.max_ulp, steps)
    if abs_err <= tol.atol + tol.rtol * abs(ref) or (tol.ulp is not None and steps <= tol.ulp):
        return
    flushed = is_subnormal(ref, width) and cand == 0.0
    if flushed:
        report.subnormal_flushed += 1
    why = "subnormal reference became zero" if flushed else "outside tolerance"
    report.note(place, ref, cand, why, {"abs_err": abs_err, "rel_err": rel_err, "ulp": steps})


def compare_series(
    name: str, tol: Tolerance | None, ref: Series, cand: Series | None
) -> FieldReport:
    report = FieldReport(name, tol, count=len(ref.values))
    if cand is None:
        report.structure = "missing from the candidate"
        return report
    report.structure = shape_gap(name, ref.nodes, cand.nodes)
    if report.structure is not None:
        return report
    for (place, left), (_, right) in zip(ref.values, cand.values):
        if exact_pair(left, right):
            if type(left) is not type(right) or left != right:
                report.int_mismatch += 1
                report.note(place, left, right, "integer values differ")
            continue
        if tol is None:
            reject(
                f"The field {name} has floats and the tolerance file does not cover it. "
                "Add a default tolerance and run the compare again."
            )
        compare_float(report, tol, place, float(left), float(right))
    return report


def measure(
    ref_doc: object, cand_doc: object, cfg_doc: object
) -> tuple[bool, list[FieldReport]]:
    default, named = parse_config(cfg_doc)
    ref = collect_fields(ref_doc, "reference")
    cand = collect_fields(cand_doc, "candidate")
    unused = sorted(set(named) - set(ref))
    if unused:
        reject(
            f"The tolerance file names {unused}, which the reference does not have. "
            "Remove that name and run the compare again."
        )
    reports = [
        compare_series(name, named.get(name, default), series, cand.get(name))
        for name, series in ref.items()
    ]
    for name, series in cand.items():
        if name in ref:
            continue
        extra = FieldReport(name, None, count=len(series.values))
        extra.structure = "extra field: the candidate has it and the reference does not"
        reports.append(extra)
    return all(item.passed() for item in reports), reports


def json_safe(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def build_report(
    ref: Path, cand: Path, config: Path, passed: bool, reports: list[FieldReport]
) -> dict[str, object]:
    comparable = [item for item in reports if item.structure is None]
    summary = {
        "fields": len(reports),
        "fields_failed": sum(not item.passed() for item in reports),
        "values": sum(item.count for item in reports),
        "failures": sum(item.failures for item in reports),
        "max_rel_err": max((item.max_rel_err for item in comparable), default=0.0),
        "max_ulp": max((item.max_ulp for item in comparable), default=0),
    }
    return {
        "tool": "parity",
        "version": VERSION,
        "pass": passed,
        "inputs": {
            "reference": {"path": stored_path(ref), "sha256": file_sha256(ref)},
            "candidate": {"path": stored_path(cand), "sha256": file_sha256(cand)},
            "config": {"path": stored_path(config), "sha256": file_sha256(config)},
        },
        "summary": summary,
        "fields": [item.as_dict() for item in reports],
    }


def same_resolved_path(left: Path, right: Path) -> bool:
    # A link or a ".." segment must not hide that the report replaces an input.
    return left.resolve() == right.resolve()


def reject_report_alias(report: Path, ref: Path, cand: Path, config: Path) -> None:
    named = (("reference", ref), ("candidate", cand), ("config", config))
    for role, path in named:
        if same_resolved_path(report, path):
            reject(
                f"The report path resolves to the {role} file. "
                "Choose a different report path and run the compare again."
            )


def print_summary(report: dict[str, object]) -> None:
    summary = report["summary"]
    print("summary " + json.dumps(summary, sort_keys=True))
    fields = report["fields"]
    if not isinstance(fields, list):
        return
    for item in fields:
        if not isinstance(item, dict) or item.get("pass") is True:
            continue
        name = item["name"]
        if "structure" in item:
            print(f"FAIL {name} {item['structure']}")
            continue
        print(f"FAIL {name} max_ulp {item['max_ulp']} max_rel_err {item['max_rel_err']}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="parity_compare")
    parser.add_argument("--ref", type=Path, required=True)
    parser.add_argument("--cand", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    reject_report_alias(args.report, args.ref, args.cand, args.config)
    passed, reports = measure(
        load_document(args.ref, "reference"),
        load_document(args.cand, "candidate"),
        load_document(args.config, "config"),
    )
    # No numeric value means the compare measured nothing. That must not pass.
    if sum(item.count for item in reports) == 0:
        passed = False
        print(
            "parity_compare: The comparison holds zero numeric values. "
            "Add numeric values and run the compare again.",
            file=sys.stderr,
        )
    report = json_safe(build_report(args.ref, args.cand, args.config, passed, reports))
    if not isinstance(report, dict):
        reject("The report is not an object. Run the compare again.")
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    try:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    except OSError as exc:
        detail = exc.strerror or "the write failed"
        reject(
            f"Cannot write the report {args.report}. {detail}. "
            "Choose a writable path and run the compare again."
        )
    print_summary(report)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
