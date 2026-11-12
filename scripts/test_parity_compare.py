#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Check the parity comparator guards."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
COMPARE = ROOT / "parity_compare.py"
EXPORT = ROOT / "parity_export.py"
HUGE = "1" + ("0" * 309)


class ParityCompareTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def write_text(self, name: str, text: str) -> Path:
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return path

    def write_json(self, name: str, payload: object) -> Path:
        return self.write_text(name, json.dumps(payload) + "\n")

    def tolerance(self, name: str, **extra: object) -> Path:
        body: dict[str, object] = {
            "rtol": 0,
            "atol": 0,
            "reason": "Exact compare for this check.",
        }
        body.update(extra)
        return self.write_json(name, {"default": body})

    def run_script(self, script: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(script), *args],
            check=False,
            capture_output=True,
            text=True,
        )

    def compare(
        self, ref: Path, cand: Path, config: Path, report: Path
    ) -> subprocess.CompletedProcess[str]:
        return self.run_script(
            COMPARE,
            [
                "--ref",
                str(ref),
                "--cand",
                str(cand),
                "--config",
                str(config),
                "--report",
                str(report),
            ],
        )

    def load_report(self, path: Path) -> dict[str, object]:
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            self.fail("The report is not an object.")
        return document

    def test_zero_numeric_values_fail(self) -> None:
        """Check that an empty comparison fails and an empty suite stops."""
        config = self.tolerance("tol.json")
        empty_ref = self.write_json("empty_ref.json", {})
        empty_cand = self.write_json("empty_cand.json", {})
        empty = self.compare(empty_ref, empty_cand, config, self.root / "empty_report.json")
        self.assertEqual(empty.returncode, 1, empty.stderr)
        self.assertIn("zero numeric values", empty.stderr)
        self.assertNotIn("Traceback", empty.stderr)

        arrays_ref = self.write_json("arrays_ref.json", {"a": []})
        arrays_cand = self.write_json("arrays_cand.json", {"a": []})
        arrays = self.compare(arrays_ref, arrays_cand, config, self.root / "arrays_report.json")
        self.assertEqual(arrays.returncode, 1, arrays.stderr)
        self.assertIn("zero numeric values", arrays.stderr)

        mixed_ref = self.write_json("mixed_ref.json", {"a": [], "b": [1.0]})
        mixed_cand = self.write_json("mixed_cand.json", {"a": [], "b": [1.0]})
        mixed = self.compare(mixed_ref, mixed_cand, config, self.root / "mixed_report.json")
        self.assertEqual(mixed.returncode, 0, mixed.stderr + mixed.stdout)

        suite = self.root / "suite"
        suite.mkdir()
        (suite / "notes.txt").write_text("no model here\n", encoding="utf-8")
        exported = self.run_script(EXPORT, [str(suite)])
        self.assertEqual(exported.returncode, 2, exported.stderr)
        self.assertIn("no model or prediction file", exported.stderr)
        self.assertNotIn("Traceback", exported.stderr)

    def test_dotted_key_is_not_a_nested_path(self) -> None:
        """Check that a dotted key and a nested key stay different fields."""
        config = self.tolerance("tol.json")
        dotted = self.write_json("dotted.json", {"x.y": 1})
        nested = self.write_json("nested.json", {"x": {"y": 1}})
        same_dot = self.compare(dotted, dotted, config, self.root / "dot_report.json")
        same_nest = self.compare(nested, nested, config, self.root / "nest_report.json")
        cross = self.compare(dotted, nested, config, self.root / "cross_report.json")
        self.assertEqual(same_dot.returncode, 0, same_dot.stderr)
        self.assertEqual(same_nest.returncode, 0, same_nest.stderr)
        self.assertEqual(cross.returncode, 1, cross.stdout + cross.stderr)
        dot_names = [item["name"] for item in self.field_list(self.root / "dot_report.json")]
        nest_names = [item["name"] for item in self.field_list(self.root / "nest_report.json")]
        self.assertEqual(dot_names, ["x\\.y"])
        self.assertEqual(nest_names, ["x.y"])
        self.assertNotEqual(dot_names, nest_names)
        both = self.write_json("both.json", {"x.y": 1, "x": {"y": 2}})
        held = self.compare(both, both, config, self.root / "both_report.json")
        self.assertEqual(held.returncode, 0, held.stderr + held.stdout)
        both_names = sorted(
            str(item["name"]) for item in self.field_list(self.root / "both_report.json")
        )
        self.assertEqual(both_names, ["x.y", "x\\.y"])

    def test_oversized_number_exits_2(self) -> None:
        """Check that a number outside f64 exits 2 and names the key."""
        ref = self.write_json("ref.json", {"scale": 1})
        cand = self.write_json("cand.json", {"scale": 1})
        tol = self.write_text(
            "huge_tol.json",
            '{"default": {"rtol": ' + HUGE + ', "atol": 0, "reason": "The limit must fit."}}\n',
        )
        bad_tol = self.compare(ref, cand, tol, self.root / "tol_report.json")
        self.assertEqual(bad_tol.returncode, 2, bad_tol.stderr)
        self.assertIn("rtol", bad_tol.stderr)
        self.assertNotIn("Traceback", bad_tol.stderr)
        self.assertFalse((self.root / "tol_report.json").exists())

        huge_ref = self.write_text("huge_ref.json", '{"scale": ' + HUGE + "}\n")
        config = self.tolerance("tol.json")
        bad_value = self.compare(huge_ref, cand, config, self.root / "value_report.json")
        self.assertEqual(bad_value.returncode, 2, bad_value.stderr)
        self.assertIn("scale", bad_value.stderr)
        self.assertNotIn("Traceback", bad_value.stderr)

        overflow = self.write_text("overflow.json", '{"scale": 1e309}\n')
        bad_float = self.compare(overflow, overflow, config, self.root / "float_report.json")
        self.assertEqual(bad_float.returncode, 2, bad_float.stderr)
        self.assertIn("scale", bad_float.stderr)
        self.assertNotIn("Traceback", bad_float.stderr)

    def test_f32_finite_overflow_is_not_infinity(self) -> None:
        """Check that a finite f32 overflow is not the same as infinity."""
        config = self.tolerance("tol.json", precision="f32")
        finite = self.write_json("finite.json", {"a": 1e308})
        infinite = self.write_json("infinite.json", {"a": "Infinity"})
        both_inf = self.write_json("both_inf.json", {"a": "Infinity"})
        mismatch = self.compare(finite, infinite, config, self.root / "f32_report.json")
        self.assertEqual(mismatch.returncode, 1, mismatch.stdout + mismatch.stderr)
        field = self.field_list(self.root / "f32_report.json")[0]
        self.assertEqual(field["nonfinite_mismatch"], 1)
        self.assertIs(field["pass"], False)
        matched = self.compare(infinite, both_inf, config, self.root / "inf_report.json")
        self.assertEqual(matched.returncode, 0, matched.stderr + matched.stdout)

    def test_zero_reference_records_relative_error(self) -> None:
        """Check that a zero reference stores an infinite relative error."""
        config = self.tolerance("tol.json")
        ref = self.write_json("ref.json", {"a": [0.0]})
        cand = self.write_json("cand.json", {"a": [1.0]})
        result = self.compare(ref, cand, config, self.root / "zero_report.json")
        self.assertEqual(result.returncode, 1, result.stderr)
        text = (self.root / "zero_report.json").read_text(encoding="utf-8")
        report = json.loads(text)
        self.assertIsInstance(report, dict)
        field = self.field_list(self.root / "zero_report.json")[0]
        summary = report["summary"]
        self.assertIsInstance(summary, dict)
        self.assertEqual(field["max_rel_err"], "Infinity")
        self.assertEqual(field["worst_rel_index"], "[0]")
        self.assertEqual(summary["max_rel_err"], "Infinity")
        self.assertIn('"Infinity"', text)
        self.assertNotIn(": Infinity", text)

    def test_report_path_does_not_replace_an_input(self) -> None:
        """Check that the report path cannot resolve to an input."""
        ref = self.write_json("ref.json", {"a": 1})
        cand = self.write_json("cand.json", {"a": 1})
        config = self.tolerance("tol.json")
        before = ref.read_bytes()
        cases = (
            ("reference", ref),
            ("candidate", cand),
            ("config", config),
        )
        for role, path in cases:
            result = self.compare(ref, cand, config, path)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn(role, result.stderr)
            self.assertNotIn("Traceback", result.stderr)
        self.assertEqual(ref.read_bytes(), before)
        self.assertEqual(json.loads(cand.read_text(encoding="utf-8")), {"a": 1})

        link = self.root / "alias.json"
        link.symlink_to(ref)
        linked = self.compare(ref, cand, config, link)
        self.assertEqual(linked.returncode, 2, linked.stderr)
        self.assertIn("reference", linked.stderr)
        self.assertEqual(ref.read_bytes(), before)

        nested = self.root / "nested"
        nested.mkdir()
        indirect = nested / ".." / "ref.json"
        via_parent = self.compare(ref, cand, config, indirect)
        self.assertEqual(via_parent.returncode, 2, via_parent.stderr)
        self.assertEqual(ref.read_bytes(), before)

    def test_one_ulp_distance(self) -> None:
        """Check that one ULP is measured and that the ULP limit applies."""
        base = 1.0
        nxt = math.nextafter(base, math.inf)
        ref = self.write_json("ref.json", {"a": [base]})
        cand = self.write_json("cand.json", {"a": [nxt]})
        loose = self.tolerance("loose.json", ulp=1)
        tight = self.tolerance("tight.json", ulp=0)
        inside = self.compare(ref, cand, loose, self.root / "loose_report.json")
        outside = self.compare(ref, cand, tight, self.root / "tight_report.json")
        self.assertEqual(inside.returncode, 0, inside.stderr + inside.stdout)
        self.assertEqual(outside.returncode, 1, outside.stderr + outside.stdout)
        self.assertEqual(self.field_list(self.root / "loose_report.json")[0]["max_ulp"], 1)
        self.assertEqual(self.field_list(self.root / "tight_report.json")[0]["max_ulp"], 1)

    def test_negative_zero_against_zero(self) -> None:
        """Check that -0.0 against 0.0 is a sign mismatch."""
        config = self.tolerance("tol.json", signed_zero="strict")
        ref = self.write_text("ref.json", '{"z": "-0x0p+0"}\n')
        cand = self.write_text("cand.json", '{"z": 0.0}\n')
        result = self.compare(ref, cand, config, self.root / "sign_report.json")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        field = self.field_list(self.root / "sign_report.json")[0]
        self.assertEqual(field["sign_mismatch"], 1)
        self.assertEqual(field["failures"], 1)
        self.assertIs(field["pass"], False)

    def field_list(self, path: Path) -> list[dict[str, object]]:
        report = self.load_report(path)
        fields = report["fields"]
        if not isinstance(fields, list):
            self.fail("The report fields are not a list.")
        rows: list[dict[str, object]] = []
        for item in fields:
            if not isinstance(item, dict):
                self.fail("A report field is not an object.")
            rows.append(item)
        return rows


if __name__ == "__main__":
    unittest.main()
