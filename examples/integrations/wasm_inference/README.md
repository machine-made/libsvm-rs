# wasm_inference (wasm-bindgen + Node.js)

Benchmark `libsvm-rs` compiled to WebAssembly against C++ LIBSVM on a realistic `heart_scale` split.

This example:
1. Builds a `wasm32-unknown-unknown` module with `wasm-bindgen`.
2. Runs repeated train/predict loops in Node.js and times compute sections in-process.
3. Runs a C++ in-process benchmark harness linked directly to `vendor/libsvm/svm.cpp`.
4. Writes a `results.json` compatible with the global comparison figure.

## Run

```bash
bash examples/integrations/wasm_inference/run.sh
```

Optional environment variables:
- `WASM_WARMUP` (default: `3`)
- `WASM_RUNS` (default: `20`)
- `WASM_TRAIN_ROWS` (default: `180`)

## Outputs

- `examples/integrations/wasm_inference/output/results.json`
- `examples/integrations/wasm_inference/output/report.md`
- `examples/integrations/wasm_inference/output/wasm_vs_cpp.png`
- `examples/integrations/wasm_inference/output/wasm_node_raw.json`
- `examples/comparison.png` (global figure, updated to include this case)
