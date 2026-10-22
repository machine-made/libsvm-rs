# Differential Verification Report

Generated: 2026-06-10T19:13:22Z

## Summary

- Scope: `full`
- Tolerance policy: `differential-v3`
- Policy document: `reference/tolerance_policy.md`
- Total cases: `250`
- Pass: `237`
- Warn: `3`
- Fail: `0`
- Skip: `10`

## Failing Cases

- None

## Warning Cases

- `housing_scale_s3_t2_tuned`: targeted SVR near-parity drift: line 55: c=18.700176996949455 rust=18.69984179204142 diff=3.352e-04 rel=1.793e-05 > max(1.0e-08, 1.5e-05*scale); max_rel=5.674e-05 (line 439); max_abs=5.317e-04 (line 438); rho_rel=9.083e-06; max sv_coef abs diff 3.563e-03; cross-predict parity passed; probability outputs differ: line 55: c=18.700176996949455 rust=18.69984179204142 diff=3.352e-04 rel=1.793e-05 > max(1.0e-08, 1.5e-05*scale)
- `gen_extreme_scale_scale_s0_t1_default`: rho-only header drift: key rho[1]: c=464.66176885411807 rust=483.86153070680615 diff=1.920e+01 rel=3.968e-02 > threshold=4.839e+00; rho_rel=3.968e-02; max sv_coef abs diff 9.719e-13; probability model header differs: key rho[1]: c=464.66176885411807 rust=483.86153070680615 diff=1.920e+01 rel=3.968e-02 > threshold=4.839e+00
- `gen_extreme_scale_scale_s2_t1_default`: one-class near-boundary label drift: line 10: label mismatch c=1.0 rust=-1.0; rho_rel=1.525e-07; max sv_coef abs diff 1.437e-09
