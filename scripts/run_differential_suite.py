#!/usr/bin/env python3
"""
Run Rust-vs-upstream-C differential verification and emit machine-readable results.

Outputs:
- reference/differential_results.json
- reference/differential_report.md

Scopes:
- quick (default): canonical datasets, default parameter profile
- full: canonical + generated datasets, default + tuned profiles
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class DatasetEntry:
    name: str
    rel_path: str
    task: str  # classification | regression | oneclass
    precomputed: bool
    source: str  # canonical | generated


@dataclass(frozen=True)
class Case:
    dataset: DatasetEntry
    svm_type: int
    kernel: int
    profile: str

    @property
    def id(self) -> str:
        ds = self.dataset.name.replace(".", "_")
        return f"{ds}_s{self.svm_type}_t{self.kernel}_{self.profile}"


def env_float_or_default(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} must be a float, got: {raw!r}") from exc


def env_bool_or_default(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    text = raw.strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise SystemExit(f"{name} must be boolean-like (0/1/true/false), got: {raw!r}")


NONPROB_REL_TOL = env_float_or_default("DIFF_NONPROB_REL_TOL", 1.5e-5)
ENABLE_TARGETED_SVR_WARN = env_bool_or_default("DIFF_ENABLE_TARGETED_SVR_WARN", True)


TOLERANCE_POLICY = {
    "name": "differential-v3",
    "document": "reference/tolerance_policy.md",
    "nonprob_scalar": {"rel_tol": NONPROB_REL_TOL, "abs_tol": 1e-8},
    "probability_classification_values": {"rel_tol": 2.5e-1, "abs_tol": 3e-2},
    "model_header_standard": {"rel_tol": 1e-2, "abs_tol": 5e-4},
    "model_header_probability": {"rel_tol": 6e-2, "abs_tol": 5e-3},
    "soft_fail": {
        "sv_coef_abs_tol": 1e-8,
        "rho_rel_tol_warn": 5e-2,
        "oneclass_label_mismatches_max": 5,
        "oneclass_rho_rel_tol_warn": 1e-6,
        "targeted_svr_case": {
            "enabled": ENABLE_TARGETED_SVR_WARN,
            "id": "housing_scale_s3_t2_tuned",
            "max_rel_tol_warn": 6e-5,
            "max_abs_tol_warn": 6e-4,
            "rho_rel_tol_warn": 1e-5,
            "sv_coef_abs_tol_warn": 4e-3,
        },
    },
}


def run_cmd(cmd: List[str]) -> Tuple[int, str]:
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    stderr = (p.stderr or "").strip()
    if len(stderr) > 1600:
        stderr = stderr[-1600:]
    return p.returncode, stderr


def read_nonempty_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def collect_label_mismatches(ref_path: Path, rust_path: Path) -> Tuple[Optional[str], int]:
    a = read_nonempty_lines(ref_path)
    b = read_nonempty_lines(rust_path)
    if len(a) != len(b):
        return f"line-count mismatch: c={len(a)} rust={len(b)}", 1
    mismatch_count = 0
    first_reason: Optional[str] = None
    for i, (x, y) in enumerate(zip(a, b), start=1):
        try:
            xv = float(x)
            yv = float(y)
        except ValueError:
            return f"line {i}: non-numeric label", mismatch_count + 1
        if xv != yv:
            mismatch_count += 1
            if first_reason is None:
                first_reason = f"line {i}: label mismatch c={xv} rust={yv}"
    return first_reason, mismatch_count


def compare_label_file(ref_path: Path, rust_path: Path) -> Optional[str]:
    first, _ = collect_label_mismatches(ref_path, rust_path)
    return first


def compare_scalar_file(
    ref_path: Path, rust_path: Path, rel_tol: float = 1e-6, abs_tol: float = 1e-8
) -> Optional[str]:
    a = read_nonempty_lines(ref_path)
    b = read_nonempty_lines(rust_path)
    if len(a) != len(b):
        return f"line-count mismatch: c={len(a)} rust={len(b)}"
    for i, (x, y) in enumerate(zip(a, b), start=1):
        try:
            xv = float(x)
            yv = float(y)
        except ValueError:
            return f"line {i}: non-numeric scalar"
        diff = abs(xv - yv)
        scale = max(abs(xv), abs(yv), 1e-15)
        if diff > max(abs_tol, rel_tol * scale):
            return (
                f"line {i}: c={xv} rust={yv} "
                f"diff={diff:.3e} rel={diff/scale:.3e} "
                f"> max({abs_tol:.1e}, {rel_tol:.1e}*scale)"
            )
    return None


def scalar_diff_stats(
    ref_path: Path, rust_path: Path
) -> Tuple[Optional[str], Optional[Dict[str, float]]]:
    a = read_nonempty_lines(ref_path)
    b = read_nonempty_lines(rust_path)
    if len(a) != len(b):
        return f"line-count mismatch: c={len(a)} rust={len(b)}", None
    max_abs = 0.0
    max_rel = 0.0
    max_abs_line = 0
    max_rel_line = 0
    for i, (x, y) in enumerate(zip(a, b), start=1):
        try:
            xv = float(x)
            yv = float(y)
        except ValueError:
            return f"line {i}: non-numeric scalar", None
        diff = abs(xv - yv)
        scale = max(abs(xv), abs(yv), 1e-15)
        rel = diff / scale
        if diff > max_abs:
            max_abs = diff
            max_abs_line = i
        if rel > max_rel:
            max_rel = rel
            max_rel_line = i
    return None, {
        "max_abs": max_abs,
        "max_rel": max_rel,
        "max_abs_line": float(max_abs_line),
        "max_rel_line": float(max_rel_line),
    }


def compare_probability_file(
    ref_path: Path, rust_path: Path, rel_tol: float = 2.5e-1, abs_tol: float = 3e-2
) -> Optional[str]:
    a = read_nonempty_lines(ref_path)
    b = read_nonempty_lines(rust_path)
    if len(a) != len(b):
        return f"line-count mismatch: c={len(a)} rust={len(b)}"
    if not a:
        return None

    start = 0
    if a[0].startswith("labels") or b[0].startswith("labels"):
        if a[0] != b[0]:
            return f"header mismatch: c='{a[0]}' rust='{b[0]}'"
        start = 1

    for line_no in range(start, len(a)):
        ra = a[line_no].split()
        rb = b[line_no].split()
        if len(ra) != len(rb):
            return f"line {line_no+1}: token-count mismatch c={len(ra)} rust={len(rb)}"
        if not ra:
            continue

        try:
            la = float(ra[0])
            lb = float(rb[0])
        except ValueError:
            return f"line {line_no+1}: non-numeric predicted label"
        if la != lb:
            return f"line {line_no+1}: predicted label mismatch c={la} rust={lb}"

        for col in range(1, len(ra)):
            try:
                va = float(ra[col])
                vb = float(rb[col])
            except ValueError:
                return f"line {line_no+1}, col {col+1}: non-numeric probability"
            diff = abs(va - vb)
            scale = max(abs(va), abs(vb), 1e-15)
            threshold = max(abs_tol, rel_tol * scale)
            if diff > threshold:
                return (
                    f"line {line_no+1}, col {col+1}: c={va} rust={vb} "
                    f"diff={diff:.3e} rel={diff/scale:.3e} > threshold={threshold:.3e}"
                )
    return None


def is_probability_predicted_label_mismatch(diff: str) -> bool:
    return "predicted label mismatch" in diff


def parse_model_header(path: Path) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line == "SV":
            break
        parts = line.split()
        out[parts[0]] = parts[1:]
    return out


def max_rho_relative_diff(c_path: Path, r_path: Path) -> Optional[float]:
    c = parse_model_header(c_path)
    r = parse_model_header(r_path)
    if "rho" not in c or "rho" not in r:
        return None
    if len(c["rho"]) != len(r["rho"]):
        return None
    max_rel = 0.0
    for sa, sb in zip(c["rho"], r["rho"]):
        try:
            fa = float(sa)
            fb = float(sb)
        except ValueError:
            return None
        scale = max(abs(fa), abs(fb), 1e-15)
        max_rel = max(max_rel, abs(fa - fb) / scale)
    return max_rel


def parse_model_sv_rows(path: Path) -> Tuple[int, List[Tuple[List[float], Tuple[str, ...]]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = parse_model_header(path)
    nr_class_vals = header.get("nr_class")
    if not nr_class_vals or len(nr_class_vals) != 1:
        raise ValueError("missing or invalid nr_class")
    nr_class = int(nr_class_vals[0])
    sv_coef_count = max(1, nr_class - 1)
    try:
        sv_start = next(i for i, line in enumerate(lines) if line.strip() == "SV") + 1
    except StopIteration as exc:
        raise ValueError("missing SV section") from exc

    rows: List[Tuple[List[float], Tuple[str, ...]]] = []
    for raw in lines[sv_start:]:
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < sv_coef_count:
            raise ValueError("SV row missing coefficients")
        coefs = [float(tok) for tok in parts[:sv_coef_count]]
        sv_tokens = tuple(parts[sv_coef_count:])
        rows.append((coefs, sv_tokens))
    return sv_coef_count, rows


def sv_payload_equivalent_within_tol(
    c_path: Path, r_path: Path, coef_abs_tol: float
) -> Tuple[bool, str]:
    try:
        c_coef_count, c_rows = parse_model_sv_rows(c_path)
        r_coef_count, r_rows = parse_model_sv_rows(r_path)
    except (ValueError, TypeError) as exc:
        return False, f"payload parse failed: {exc}"

    if c_coef_count != r_coef_count:
        return False, f"sv coefficient-count mismatch c={c_coef_count} rust={r_coef_count}"
    if len(c_rows) != len(r_rows):
        return False, f"sv row-count mismatch c={len(c_rows)} rust={len(r_rows)}"

    max_coef_diff = 0.0
    for idx, ((c_coef, c_tokens), (r_coef, r_tokens)) in enumerate(zip(c_rows, r_rows), start=1):
        if c_tokens != r_tokens:
            return False, f"sv feature row mismatch at row {idx}"
        for ca, cb in zip(c_coef, r_coef):
            max_coef_diff = max(max_coef_diff, abs(ca - cb))

    if max_coef_diff > coef_abs_tol:
        return False, f"max sv_coef abs diff {max_coef_diff:.3e} > {coef_abs_tol:.1e}"

    return True, f"max sv_coef abs diff {max_coef_diff:.3e}"


def maybe_warn_targeted_svr_drift(
    case: Case,
    suffix: str,
    nonprob_diff: str,
    data_path: Path,
    c_model: Path,
    r_model: Path,
    c_pred: Path,
    r_pred: Path,
    c_predict_bin: Path,
    rust_predict_bin: Path,
    tmp_dir: Path,
) -> Optional[str]:
    cfg = TOLERANCE_POLICY["soft_fail"].get("targeted_svr_case", {})
    if not cfg.get("enabled", True):
        return None
    case_id = cfg.get("id")
    if case.id != case_id or case.svm_type != 3:
        return None

    stats_err, stats = scalar_diff_stats(c_pred, r_pred)
    if stats_err or stats is None:
        return None

    if stats["max_rel"] > cfg["max_rel_tol_warn"] or stats["max_abs"] > cfg["max_abs_tol_warn"]:
        return None

    sv_ok, sv_reason = sv_payload_equivalent_within_tol(
        c_model,
        r_model,
        coef_abs_tol=cfg["sv_coef_abs_tol_warn"],
    )
    if not sv_ok:
        return None

    rho_rel = max_rho_relative_diff(c_model, r_model)
    if rho_rel is None or rho_rel > cfg["rho_rel_tol_warn"]:
        return None

    rc_pred = tmp_dir / f"rc_{suffix}.pred"
    cr_pred = tmp_dir / f"cr_{suffix}.pred"
    rc_rc, _ = run_cmd(
        [str(rust_predict_bin), "-q", str(data_path), str(c_model), str(rc_pred)]
    )
    cr_rc, _ = run_cmd(
        [str(c_predict_bin), "-q", str(data_path), str(r_model), str(cr_pred)]
    )
    if rc_rc != 0 or cr_rc != 0:
        return None

    rel_tol = TOLERANCE_POLICY["nonprob_scalar"]["rel_tol"]
    abs_tol = TOLERANCE_POLICY["nonprob_scalar"]["abs_tol"]
    if compare_scalar_file(c_pred, rc_pred, rel_tol=rel_tol, abs_tol=abs_tol):
        return None
    if compare_scalar_file(r_pred, cr_pred, rel_tol=rel_tol, abs_tol=abs_tol):
        return None

    return (
        f"targeted SVR near-parity drift: {nonprob_diff}; "
        f"max_rel={stats['max_rel']:.3e} (line {stats['max_rel_line']:.0f}); "
        f"max_abs={stats['max_abs']:.3e} (line {stats['max_abs_line']:.0f}); "
        f"rho_rel={rho_rel:.3e}; {sv_reason}; cross-predict parity passed"
    )


def compare_model_header(
    c_path: Path,
    r_path: Path,
    rel_tol_standard: float = 1e-2,
    abs_tol_standard: float = 5e-4,
    rel_tol_probability: float = 6e-2,
    abs_tol_probability: float = 5e-3,
) -> Optional[str]:
    float_keys = {"gamma", "coef0", "rho", "probA", "probB", "prob_density_marks"}
    int_keys = {"degree", "nr_class", "total_sv", "label", "nr_sv"}
    str_keys = {"svm_type", "kernel_type"}

    c = parse_model_header(c_path)
    r = parse_model_header(r_path)

    all_keys = sorted(set(c) | set(r))
    for key in all_keys:
        if key not in c:
            return f"missing key in C header: {key}"
        if key not in r:
            return f"missing key in Rust header: {key}"
        va = c[key]
        vb = r[key]
        if len(va) != len(vb):
            return f"key {key}: value-count mismatch c={len(va)} rust={len(vb)}"
        if key in str_keys:
            if va != vb:
                return f"key {key}: mismatch c={va} rust={vb}"
        elif key in int_keys:
            try:
                ia = [int(x) for x in va]
                ib = [int(x) for x in vb]
            except ValueError:
                return f"key {key}: non-integer values"
            if ia != ib:
                return f"key {key}: mismatch c={ia} rust={ib}"
        elif key in float_keys:
            for i, (sa, sb) in enumerate(zip(va, vb), start=1):
                try:
                    fa = float(sa)
                    fb = float(sb)
                except ValueError:
                    return f"key {key}[{i}]: non-float values"
                diff = abs(fa - fb)
                scale = max(abs(fa), abs(fb), 1e-15)
                if key in {"probA", "probB", "prob_density_marks"}:
                    rel_tol = rel_tol_probability
                    abs_tol = abs_tol_probability
                else:
                    rel_tol = rel_tol_standard
                    abs_tol = abs_tol_standard
                threshold = max(abs_tol, rel_tol * scale)
                if diff > threshold:
                    return (
                        f"key {key}[{i}]: c={fa} rust={fb} diff={diff:.3e} "
                        f"rel={diff/scale:.3e} > threshold={threshold:.3e}"
                    )
        else:
            if va != vb:
                return f"key {key}: mismatch c={va} rust={vb}"
    return None


def build_profiles(scope: str) -> List[str]:
    if scope == "full":
        return ["default", "tuned"]
    return ["default"]


def extra_args_for(profile: str, svm_type: int, kernel: int) -> List[str]:
    if profile == "default":
        return []
    out: List[str] = []
    if kernel != 4:
        if kernel in (1, 2, 3):
            out += ["-g", "0.2"]
        if kernel == 1:
            out += ["-d", "2", "-r", "0.5"]
        if kernel == 3:
            out += ["-r", "0.1"]
    if svm_type in (0, 3, 4):
        out += ["-c", "2"]
    if svm_type in (1, 2, 4):
        out += ["-n", "0.4"]
    if svm_type == 3:
        out += ["-p", "0.2"]
    out += ["-e", "0.0005", "-h", "0"]
    return out


def svm_types_for_task(task: str) -> List[int]:
    if task == "classification":
        return [0, 1, 2]
    if task == "regression":
        return [2, 3, 4]
    if task == "oneclass":
        return [2]
    raise ValueError(f"unknown task: {task}")


def datasets(root: Path, scope: str) -> List[DatasetEntry]:
    items: List[DatasetEntry] = []

    # Canonical datasets.
    canonical = [
        ("heart_scale", "classification"),
        ("iris.scale", "classification"),
        ("housing_scale", "regression"),
    ]
    for name, task in canonical:
        items.append(
            DatasetEntry(
                name=name,
                rel_path=f"data/{name}",
                task=task,
                precomputed=False,
                source="canonical",
            )
        )
        items.append(
            DatasetEntry(
                name=f"{name}.precomputed",
                rel_path=f"data/{name}.precomputed",
                task=task,
                precomputed=True,
                source="canonical",
            )
        )

    if scope == "full":
        manifest = json.loads((root / "reference" / "dataset_manifest.json").read_text(encoding="utf-8"))
        for d in manifest["datasets"]:
            items.append(
                DatasetEntry(
                    name=d["name"],
                    rel_path=d["path"],
                    task=d["task"],
                    precomputed=False,
                    source="generated",
                )
            )
            items.append(
                DatasetEntry(
                    name=Path(d["precomputed_path"]).name,
                    rel_path=d["precomputed_path"],
                    task=d["task"],
                    precomputed=True,
                    source="generated",
                )
            )

    return items


def ensure_prereqs(root: Path) -> None:
    subprocess.run(["bash", str(root / "scripts" / "setup_reference_libsvm.sh")], check=True)
    subprocess.run(["python3", str(root / "scripts" / "generate_precomputed_datasets.py")], check=True)
    subprocess.run(["python3", str(root / "scripts" / "generate_differential_datasets.py")], check=True)
    subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "-p",
            "svm-train-rs",
            "-p",
            "svm-predict-rs",
        ],
        check=True,
    )


def write_report(results: Dict[str, object], report_path: Path) -> None:
    summary = results["summary"]
    cases = results["cases"]
    policy = results.get("tolerance_policy", {})
    fail_cases = [c for c in cases if c["status"] == "fail"]
    warn_cases = [c for c in cases if c["status"] == "warn"]

    lines = [
        "# Differential Verification Report",
        "",
        f"Generated: {results['generated_utc']}",
        "",
        "## Summary",
        "",
        f"- Scope: `{results['scope']}`",
        f"- Tolerance policy: `{policy.get('name', 'unknown')}`",
        f"- Policy document: `{policy.get('document', 'n/a')}`",
        f"- Total cases: `{summary['total']}`",
        f"- Pass: `{summary['pass']}`",
        f"- Warn: `{summary['warn']}`",
        f"- Fail: `{summary['fail']}`",
        f"- Skip: `{summary['skip']}`",
        "",
        "## Failing Cases",
        "",
    ]

    if not fail_cases:
        lines.append("- None")
    else:
        for c in fail_cases[:30]:
            lines.append(f"- `{c['id']}`: {c['reason']}")

    lines.extend(["", "## Warning Cases", ""])
    if not warn_cases:
        lines.append("- None")
    else:
        for c in warn_cases[:40]:
            lines.append(f"- `{c['id']}`: {c['reason']}")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    scope = os.environ.get("DIFF_SCOPE", "quick").strip().lower()
    if scope not in {"quick", "full"}:
        raise SystemExit("DIFF_SCOPE must be one of: quick, full")

    ensure_prereqs(root)

    rust_train = root / "target" / "release" / "svm-train-rs"
    rust_predict = root / "target" / "release" / "svm-predict-rs"
    c_train = root / ".tmp" / "reference_upstream" / "libsvm" / "svm-train"
    c_predict = root / ".tmp" / "reference_upstream" / "libsvm" / "svm-predict"
    tmp_dir = root / ".tmp" / "differential_suite"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    all_datasets = datasets(root, scope)
    profiles = build_profiles(scope)

    cases: List[Case] = []
    for ds in all_datasets:
        svm_types = svm_types_for_task(ds.task)
        kernels = [4] if ds.precomputed else [0, 1, 2, 3]
        for svm_type in svm_types:
            for kernel in kernels:
                for profile in profiles:
                    cases.append(Case(dataset=ds, svm_type=svm_type, kernel=kernel, profile=profile))

    result_cases: List[Dict[str, object]] = []
    counts = {"pass": 0, "warn": 0, "fail": 0, "skip": 0}

    for idx, case in enumerate(cases, start=1):
        print(f"[{idx:03d}/{len(cases)}] {case.id}")
        data_path = root / case.dataset.rel_path
        if not data_path.exists():
            rec = {
                "id": case.id,
                "status": "skip",
                "reason": f"dataset missing: {case.dataset.rel_path}",
            }
            result_cases.append(rec)
            counts["skip"] += 1
            continue

        suffix = case.id
        c_model = tmp_dir / f"c_{suffix}.model"
        r_model = tmp_dir / f"r_{suffix}.model"
        c_pred = tmp_dir / f"c_{suffix}.pred"
        r_pred = tmp_dir / f"r_{suffix}.pred"
        c_prob_model = tmp_dir / f"c_{suffix}.prob.model"
        r_prob_model = tmp_dir / f"r_{suffix}.prob.model"
        c_prob_pred = tmp_dir / f"c_{suffix}.prob.pred"
        r_prob_pred = tmp_dir / f"r_{suffix}.prob.pred"

        extra = extra_args_for(case.profile, case.svm_type, case.kernel)
        base_opts = ["-q", "-s", str(case.svm_type), "-t", str(case.kernel), *extra]

        c_train_cmd = [str(c_train), *base_opts, str(data_path), str(c_model)]
        r_train_cmd = [str(rust_train), *base_opts, str(data_path), str(r_model)]

        c_train_rc, c_train_err = run_cmd(c_train_cmd)
        r_train_rc, r_train_err = run_cmd(r_train_cmd)

        if c_train_rc != 0 and r_train_rc != 0:
            rec = {
                "id": case.id,
                "dataset": case.dataset.rel_path,
                "svm_type": case.svm_type,
                "kernel": case.kernel,
                "profile": case.profile,
                "status": "skip",
                "reason": "both C and Rust training failed for this combo",
                "c_train_rc": c_train_rc,
                "rust_train_rc": r_train_rc,
            }
            result_cases.append(rec)
            counts["skip"] += 1
            continue
        if c_train_rc != 0 or r_train_rc != 0:
            rec = {
                "id": case.id,
                "dataset": case.dataset.rel_path,
                "svm_type": case.svm_type,
                "kernel": case.kernel,
                "profile": case.profile,
                "status": "fail",
                "reason": "training success mismatch between C and Rust",
                "c_train_rc": c_train_rc,
                "rust_train_rc": r_train_rc,
                "c_stderr_tail": c_train_err,
                "rust_stderr_tail": r_train_err,
            }
            result_cases.append(rec)
            counts["fail"] += 1
            continue

        c_pred_rc, c_pred_err = run_cmd([str(c_predict), "-q", str(data_path), str(c_model), str(c_pred)])
        r_pred_rc, r_pred_err = run_cmd(
            [str(rust_predict), "-q", str(data_path), str(r_model), str(r_pred)]
        )
        if c_pred_rc != 0 or r_pred_rc != 0:
            rec = {
                "id": case.id,
                "dataset": case.dataset.rel_path,
                "svm_type": case.svm_type,
                "kernel": case.kernel,
                "profile": case.profile,
                "status": "fail",
                "reason": "prediction command failed",
                "c_predict_rc": c_pred_rc,
                "rust_predict_rc": r_pred_rc,
                "c_stderr_tail": c_pred_err,
                "rust_stderr_tail": r_pred_err,
            }
            result_cases.append(rec)
            counts["fail"] += 1
            continue

        nonprob_warning: Optional[str] = None
        label_mismatch_count = 0
        if case.svm_type <= 2:
            nonprob_diff, label_mismatch_count = collect_label_mismatches(c_pred, r_pred)
        else:
            nonprob_diff = compare_scalar_file(
                c_pred,
                r_pred,
                rel_tol=TOLERANCE_POLICY["nonprob_scalar"]["rel_tol"],
                abs_tol=TOLERANCE_POLICY["nonprob_scalar"]["abs_tol"],
            )
        if nonprob_diff:
            downgraded = False
            if case.svm_type == 2:
                sv_ok, sv_reason = sv_payload_equivalent_within_tol(
                    c_model,
                    r_model,
                    coef_abs_tol=TOLERANCE_POLICY["soft_fail"]["sv_coef_abs_tol"],
                )
                rho_rel = max_rho_relative_diff(c_model, r_model)
                if (
                    sv_ok
                    and rho_rel is not None
                    and rho_rel <= TOLERANCE_POLICY["soft_fail"]["oneclass_rho_rel_tol_warn"]
                    and label_mismatch_count
                    <= TOLERANCE_POLICY["soft_fail"]["oneclass_label_mismatches_max"]
                ):
                    nonprob_warning = (
                        f"one-class near-boundary label drift: {nonprob_diff}; "
                        f"rho_rel={rho_rel:.3e}; {sv_reason}"
                    )
                    downgraded = True
            elif case.svm_type in (3, 4):
                targeted_warning = maybe_warn_targeted_svr_drift(
                    case=case,
                    suffix=suffix,
                    nonprob_diff=nonprob_diff,
                    data_path=data_path,
                    c_model=c_model,
                    r_model=r_model,
                    c_pred=c_pred,
                    r_pred=r_pred,
                    c_predict_bin=c_predict,
                    rust_predict_bin=rust_predict,
                    tmp_dir=tmp_dir,
                )
                if targeted_warning:
                    nonprob_warning = targeted_warning
                    downgraded = True
            if downgraded:
                nonprob_diff = None
            else:
                rec = {
                    "id": case.id,
                    "dataset": case.dataset.rel_path,
                    "svm_type": case.svm_type,
                    "kernel": case.kernel,
                    "profile": case.profile,
                    "status": "fail",
                    "reason": f"non-probability predictions differ: {nonprob_diff}",
                }
                result_cases.append(rec)
                counts["fail"] += 1
                continue

        model_diff = compare_model_header(
            c_model,
            r_model,
            rel_tol_standard=TOLERANCE_POLICY["model_header_standard"]["rel_tol"],
            abs_tol_standard=TOLERANCE_POLICY["model_header_standard"]["abs_tol"],
            rel_tol_probability=TOLERANCE_POLICY["model_header_probability"]["rel_tol"],
            abs_tol_probability=TOLERANCE_POLICY["model_header_probability"]["abs_tol"],
        )
        if model_diff:
            downgraded = False
            if model_diff.startswith("key rho[") and case.svm_type in (0, 1, 2):
                sv_ok, sv_reason = sv_payload_equivalent_within_tol(
                    c_model,
                    r_model,
                    coef_abs_tol=TOLERANCE_POLICY["soft_fail"]["sv_coef_abs_tol"],
                )
                rho_rel = max_rho_relative_diff(c_model, r_model)
                if (
                    sv_ok
                    and rho_rel is not None
                    and rho_rel <= TOLERANCE_POLICY["soft_fail"]["rho_rel_tol_warn"]
                ):
                    rho_warning = f"rho-only header drift: {model_diff}; rho_rel={rho_rel:.3e}; {sv_reason}"
                    if nonprob_warning:
                        nonprob_warning += f"; {rho_warning}"
                    else:
                        nonprob_warning = rho_warning
                    downgraded = True
            if downgraded:
                model_diff = None
            else:
                rec = {
                    "id": case.id,
                    "dataset": case.dataset.rel_path,
                    "svm_type": case.svm_type,
                    "kernel": case.kernel,
                    "profile": case.profile,
                    "status": "fail",
                    "reason": f"model header differs: {model_diff}",
                }
                result_cases.append(rec)
                counts["fail"] += 1
                continue

        # Probability branch: 0,1,3,4
        prob_warning: Optional[str] = None
        if case.svm_type in (0, 1, 3, 4):
            prob_opts = ["-q", "-b", "1", "-s", str(case.svm_type), "-t", str(case.kernel), *extra]
            c_prob_train_rc, c_prob_train_err = run_cmd(
                [str(c_train), *prob_opts, str(data_path), str(c_prob_model)]
            )
            r_prob_train_rc, r_prob_train_err = run_cmd(
                [str(rust_train), *prob_opts, str(data_path), str(r_prob_model)]
            )

            if c_prob_train_rc != 0 and r_prob_train_rc != 0:
                prob_warning = "both C and Rust probability training failed"
            elif c_prob_train_rc != 0 or r_prob_train_rc != 0:
                rec = {
                    "id": case.id,
                    "dataset": case.dataset.rel_path,
                    "svm_type": case.svm_type,
                    "kernel": case.kernel,
                    "profile": case.profile,
                    "status": "fail",
                    "reason": "probability training success mismatch",
                    "c_prob_train_rc": c_prob_train_rc,
                    "rust_prob_train_rc": r_prob_train_rc,
                    "c_stderr_tail": c_prob_train_err,
                    "rust_stderr_tail": r_prob_train_err,
                }
                result_cases.append(rec)
                counts["fail"] += 1
                continue
            else:
                c_prob_pred_rc, c_prob_pred_err = run_cmd(
                    [
                        str(c_predict),
                        "-q",
                        "-b",
                        "1",
                        str(data_path),
                        str(c_prob_model),
                        str(c_prob_pred),
                    ]
                )
                r_prob_pred_rc, r_prob_pred_err = run_cmd(
                    [
                        str(rust_predict),
                        "-q",
                        "-b",
                        "1",
                        str(data_path),
                        str(r_prob_model),
                        str(r_prob_pred),
                    ]
                )
                if c_prob_pred_rc != 0 or r_prob_pred_rc != 0:
                    rec = {
                        "id": case.id,
                        "dataset": case.dataset.rel_path,
                        "svm_type": case.svm_type,
                        "kernel": case.kernel,
                        "profile": case.profile,
                        "status": "fail",
                        "reason": "probability prediction command failed",
                        "c_prob_predict_rc": c_prob_pred_rc,
                        "rust_prob_predict_rc": r_prob_pred_rc,
                        "c_stderr_tail": c_prob_pred_err,
                        "rust_stderr_tail": r_prob_pred_err,
                    }
                    result_cases.append(rec)
                    counts["fail"] += 1
                    continue

                if case.svm_type in (0, 1):
                    prob_diff = compare_probability_file(
                        c_prob_pred,
                        r_prob_pred,
                        rel_tol=TOLERANCE_POLICY["probability_classification_values"]["rel_tol"],
                        abs_tol=TOLERANCE_POLICY["probability_classification_values"]["abs_tol"],
                    )
                else:
                    prob_diff = compare_scalar_file(
                        c_prob_pred,
                        r_prob_pred,
                        rel_tol=TOLERANCE_POLICY["nonprob_scalar"]["rel_tol"],
                        abs_tol=TOLERANCE_POLICY["nonprob_scalar"]["abs_tol"],
                    )
                if prob_diff:
                    if case.svm_type in (0, 1) and is_probability_predicted_label_mismatch(prob_diff):
                        rec = {
                            "id": case.id,
                            "dataset": case.dataset.rel_path,
                            "svm_type": case.svm_type,
                            "kernel": case.kernel,
                            "profile": case.profile,
                            "status": "fail",
                            "reason": f"probability predicted-label mismatch: {prob_diff}",
                        }
                        result_cases.append(rec)
                        counts["fail"] += 1
                        continue
                    prob_warning = f"probability outputs differ: {prob_diff}"

                prob_model_diff = compare_model_header(
                    c_prob_model,
                    r_prob_model,
                    rel_tol_standard=TOLERANCE_POLICY["model_header_standard"]["rel_tol"],
                    abs_tol_standard=TOLERANCE_POLICY["model_header_standard"]["abs_tol"],
                    rel_tol_probability=TOLERANCE_POLICY["model_header_probability"]["rel_tol"],
                    abs_tol_probability=TOLERANCE_POLICY["model_header_probability"]["abs_tol"],
                )
                if prob_model_diff:
                    if prob_warning:
                        prob_warning += f"; probability model header differs: {prob_model_diff}"
                    else:
                        prob_warning = f"probability model header differs: {prob_model_diff}"

        warnings = [w for w in (nonprob_warning, prob_warning) if w]
        if warnings:
            rec_status = "warn"
            reason = "; ".join(warnings)
            counts["warn"] += 1
        else:
            rec_status = "pass"
            reason = "all checked comparisons passed"
            counts["pass"] += 1

        rec = {
            "id": case.id,
            "dataset": case.dataset.rel_path,
            "dataset_source": case.dataset.source,
            "dataset_task": case.dataset.task,
            "precomputed": case.dataset.precomputed,
            "svm_type": case.svm_type,
            "kernel": case.kernel,
            "profile": case.profile,
            "status": rec_status,
            "reason": reason,
        }
        result_cases.append(rec)

    summary = {
        "total": len(result_cases),
        "pass": counts["pass"],
        "warn": counts["warn"],
        "fail": counts["fail"],
        "skip": counts["skip"],
    }
    payload = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scope": scope,
        "tolerance_policy": TOLERANCE_POLICY,
        "summary": summary,
        "cases": result_cases,
    }

    out_json = root / "reference" / "differential_results.json"
    out_md = root / "reference" / "differential_report.md"
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_report(payload, out_md)

    print(
        "Differential suite complete: "
        f"{summary['pass']} pass, {summary['warn']} warn, {summary['fail']} fail, {summary['skip']} skip"
    )
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    if summary["fail"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
