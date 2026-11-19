#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Check the reading of the cross-validation line of a trainer."""

from __future__ import annotations

import unittest

from run_differential_suite import read_cv_accuracy

CASE = "single_member_scale_s0_t0_default_cv3"
TRAINER = "svm-train-rs"


class ReadCvAccuracyTests(unittest.TestCase):
    def read(
        self, stdout: str, returncode: int = 0, stderr: str = ""
    ) -> tuple[str | None, str | None]:
        return read_cv_accuracy(CASE, TRAINER, returncode, stdout, stderr)

    def test_valid_line_gives_the_number_as_text(self) -> None:
        self.assertEqual(
            self.read("Cross Validation Accuracy = 88.8889%\n"), ("88.8889", None)
        )

    def test_invalid_number_gives_a_diagnostic(self) -> None:
        value, error = self.read("Cross Validation Accuracy = invalid%\n")
        self.assertIsNone(value)
        self.assertIsNotNone(error)
        self.assertIn(CASE, str(error))
        self.assertIn(TRAINER, str(error))
        self.assertIn("invalid%", str(error))

    def test_empty_number_gives_a_diagnostic(self) -> None:
        value, error = self.read("Cross Validation Accuracy = %")
        self.assertIsNone(value)
        self.assertIn(CASE, str(error))

    def test_other_output_gives_a_diagnostic(self) -> None:
        for stdout in ("", "Cross Validation Accuracy = 88.8889", "accuracy = 88%"):
            with self.subTest(stdout=stdout):
                value, error = self.read(stdout)
                self.assertIsNone(value)
                self.assertIn(TRAINER, str(error))

    def test_nonzero_exit_gives_a_diagnostic_with_stderr(self) -> None:
        value, error = self.read(
            "Cross Validation Accuracy = 88.8889%", returncode=1, stderr="bad input\n"
        )
        self.assertIsNone(value)
        self.assertIn("bad input", str(error))


if __name__ == "__main__":
    unittest.main()
