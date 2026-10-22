# Vision

**libsvm-rs is the definitive, trusted, drop-in replacement for LIBSVM in
Rust.**

When a project needs LIBSVM — its SVM types, kernels, data and model formats,
CLI workflow, or numerical behavior — and wants it in Rust without an FFI
boundary, libsvm-rs should be the obvious and safe choice.

## What "definitive replacement" means

1. **Numerical parity is the product.** Equivalence with upstream LIBSVM
   (within documented tolerances, verified by the pinned differential suite)
   outranks every other property. A change that improves anything else but
   drifts parity is a regression.
2. **Format compatibility is non-negotiable.** Problem files, model files
   (`%.17g`), scaling parameter files, and CLI flag syntax stay interchangeable
   with the C tools in both directions.
3. **Trustworthy by inspection.** Pure Rust, one runtime dependency,
   security-hardened parsers for untrusted input, pinned toolchains and
   reference builds, auditable supply chain. The library a security review
   signs off on.
4. **At least as fast as C.** Prediction already is; training should be too.
   Performance work is in scope precisely because a "replacement" that is
   slower is not a replacement.
5. **Idiomatic to hold, identical in behavior.** Ergonomic Rust APIs
   (builders, serde, dense-data helpers) are welcome as *additive* layers that
   never change semantics or break LIBSVM compatibility. New algorithmic
   capabilities upstream doesn't have are out of scope.

## Who it serves

- Rust projects replacing `libsvm`/`libsvm-sys2` FFI bindings.
- Teams porting C/C++ LIBSVM pipelines who need models and data to keep
  working unchanged.
- Deployments where the C runtime is unwanted: static binaries,
  cross-compilation, WASM inference.

## Explicit non-goals

- A general ML framework or linfa-style estimator ecosystem.
- Algorithms beyond upstream LIBSVM (online learning, GPU, new solvers).
- GUI tooling (svm-toy).
