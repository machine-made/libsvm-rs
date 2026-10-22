# Security Audit — libsvm-rs

Date: 2026-06-11 · Scope: serde deserialization surface, rayon parallel paths,
parameter-builder validation, glibc-rand LCG replication, regression check of the
2026-06-02 findings, supply chain and CI gates · Method: code review of each
surface, with reproduction where possible. Findings marked **[Verified]** were
reproduced or confirmed in the code; **[Reviewed]** findings come from code
reading only and were not reproduced.

Commit audited: `0c81b93` (main). One fix from this audit (S1) lands in the
same cycle, with a regression test.

## Summary

No High findings. The new surfaces added since the 2026-06-02 audit are sound:

- **serde** (`Deserialize` for `SvmModel`/`SvmParameter`, feature-gated): the
  critical control is the custom `Deserialize` impl on `SvmModel`
  (`types.rs`) that routes through the same `validate_model()` used by the
  text loader, so structural invariants (label/n_sv lengths, rho count,
  sv_coef shape, finiteness, ascending SV indices) hold on both paths — OOB
  panics in `predict` are not reachable via crafted JSON. Adversarial serde
  tests exist (`tests/malicious_input.rs`) and pass by rejection. One gap
  found and fixed (S1: negative gamma).
- **rayon** (parallel CV folds + probability fold trainings): no data races
  (`split_at_mut` gives exclusive fold slices), PRNG (`c_rand`) is consumed
  serially *before* any parallel region, the probability path is explicitly
  serial, and bitwise parity with the serial path is pinned by digest tests
  run by CI under three feature configurations. A suspected issue, "silent
  zero-padded CV results on fold panic", was **ruled out**: a
  panic inside the parallel region unwinds out of the call; the local result
  buffer is dropped and the caller never observes partial output.
- **Builder**: `SvmParameterBuilder::build()` delegates to
  `SvmParameter::validate()`, which matches `svm_check_parameter` (svm.cpp)
  check-for-check; the data-dependent nu-SVC feasibility check is correctly
  deferred to `check_parameter(problem, param)`. No gaps.
- **Supply chain**: deny.toml strict (yanked=deny, unmaintained=all, license
  allowlist, native-build bans), all six workflows SHA-pinned with minimal
  permissions and no script-injection vectors, dependabot covers cargo +
  github-actions, runtime deps remain minimal (thiserror; optional serde,
  rayon).

All six findings of the 2026-06-02 audit (F1–F6) were re-verified intact at
their current locations.

## Findings

### S1 — Medium — Negative gamma accepted at model load (text and serde) · [Verified]
- **Where:** `crates/libsvm/src/io.rs` `validate_model` — checked
  `gamma.is_finite()` but not sign; `SvmParameter::validate()` rejects
  `gamma < 0` for Polynomial/RBF/Sigmoid, but is not called on load.
- **What:** A model with `gamma: -1.0` deserialized (or text-loaded)
  successfully and fed a negative gamma into kernel evaluation — wrong
  predictions, not a panic. Affects both paths equally (parity gap with the
  parameter validator, not a serde-specific bypass).
- **Fix (this cycle):** `validate_model` now rejects `gamma < 0` for the
  gamma-using kernels. Regression test
  `serde_rejects_negative_gamma` (`tests/malicious_input.rs`) passes by
  rejection; full workspace suite green under `--all-features`.

### S2 — Low — `SvmParameter` deserializes without validation · [Verified]
- **Where:** `crates/libsvm/src/types.rs` — `SvmParameter` uses derived
  `Deserialize`; `validate()` is not invoked at the serde layer.
- **What:** `serde_json::from_str::<SvmParameter>()` accepts e.g.
  `cache_size <= 0`, `eps <= 0`, `nu > 1`. Harmless inside `SvmModel`
  (its custom impl validates), and training always runs
  `check_parameter()` which calls `validate()` — so invalid values are caught
  before use. Disposition: accepted; documented here. Tightening would
  require a hand-written `Deserialize` for marginal benefit.

### S3 — Low — Non-finite floats rely on the serde format's strictness · [Reviewed]
- **Where:** serde path generally; `validate_model` finiteness checks.
- **What:** serde_json rejects NaN/Inf at the parser, and `validate_model`
  independently rejects non-finite gamma/coef0/rho/sv_coef/feature values —
  so even permissive formats (bincode, msgpack) are covered for model
  fields. Parameter-only fields outside the model (see S2) are the residual.
  Disposition: accepted.

### S4 — Low — No fuzz target for the serde surface · [Verified]
- **Where:** `crates/libsvm/fuzz/fuzz_targets/` — `parse_model.rs` and
  `parse_problem.rs` cover the text parsers only.
- **What:** the serde `Deserialize` path is unit-tested adversarially but not
  fuzzed. Disposition: nice-to-have (a serde_json target is ~16 lines, ~30 s
  per scheduled fuzz run); not release-blocking since the path funnels into
  the same `validate_model` the text fuzzer already exercises.

### S5 — Informational — Fold panics propagate as panics, not `Err` · [Verified]
- **Where:** `crates/libsvm/src/cross_validation.rs` (parallel `for_each`),
  `crates/libsvm/src/probability.rs` (parallel `map/collect`).
- **What:** a panic inside a fold training unwinds through rayon to the
  caller. No corruption or partial results (verified: result
  buffers are locals dropped during unwind), and fold training on validated
  input has no known panic sites. Matches serial behavior; no change.

### S6 — Informational — glibc-rand LCG replication is concurrency-safe · [Verified]
- **Where:** `crates/libsvm/src/util.rs` (`c_rand`), call sites in
  `cross_validation.rs` / `probability.rs`.
- **What:** all PRNG consumption (shuffles, Fisher–Yates) happens serially
  before parallel regions, in the same order as the serial path — this is
  what makes the rayon/serial digest parity hold. No action.

## Historical: 2026-06-02 audit (commit `0bfe9f5`)

All findings fixed then; re-verified intact in this audit:

| ID | Sev | Status (re-verified 2026-06-11) |
|---|---|---|
| F1 | High | Intact — `validate_model_header` requires `label.len()==nr_class` and `n_sv.len()==nr_class` for c_svc/nu_svc (`io.rs:1115,1135`). |
| F2 | Med×3 | Intact — all three bins guard `flag.len() < 2` before indexing. |
| F3 | Low | Intact — non-finite gamma/coef0/rho/sv_coef/feature values and negative degree rejected at load. |
| F4 | Low | Intact — safety comment at `kernel.rs` precomputed cast. |
| F5 | Low | Intact — safety comment at `qmatrix.rs` cache-size cast. |
| F6 | Low | Intact — `deny.toml` `yanked = "deny"`, `unmaintained = "all"`. |

Full details of the original findings: see this file at commit `0bfe9f5`.

## Verification

- `cargo test --workspace --all-features` — all suites green (196 tests),
  including `serde_rejects_negative_gamma`.
- `cargo fmt --check` and
  `cargo clippy --workspace --all-targets --all-features -- -D warnings` clean.
- Rayon/serial bitwise parity: `tests/rayon_parity.rs` digests, run by CI
  under `--all-features`, `--features rayon`, and `--no-default-features`.
