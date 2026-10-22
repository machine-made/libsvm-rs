# WASM Inference Benchmark Results

Date: 2026-06-15 11:00:14Z

## Methodology

- Both runtimes use in-process timing for compute-only sections.
- Data parsing/loading is outside timing windows for both runtimes.
- Runtime A (`rust` field in JSON): Rust compiled to `wasm32-unknown-unknown`, executed in Node.js via `wasm-bindgen`.
- Runtime B (`c` field in JSON): C++ LIBSVM via an in-process benchmark harness linked to `vendor/libsvm/svm.cpp`.

## Setup

- Dataset: `heart_scale` split (classification)
- Train rows: 180
- Test rows: 90
- Parameter: C-SVC, RBF kernel, C=1, gamma=1/13
- Warmup: 3
- Runs: 20

## Timing Ratios (median)

- Train wasm/C++: 2.013
- Predict wasm/C++: 1.532

## Correctness

- WASM accuracy: 0.8444
- C++ accuracy: 0.8444
- Prediction agreement: 1.0000
