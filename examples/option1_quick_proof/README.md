# Option 1: Quick Proof (Fast)

This folder runs a compact Rust vs C++ (LIBSVM) demo on `ijcnn1` and generates curves.

## Run

```bash
bash examples/option1_quick_proof/run.sh
```

`run.sh` does both steps:
1. Downloads required reference data for Option 1.
2. Runs the Python benchmark/plot pipeline.

## Outputs

- `examples/option1_quick_proof/output/results.json`
- `examples/option1_quick_proof/output/report.md`
- `examples/option1_quick_proof/output/scientific_demo.npz`
- `examples/option1_quick_proof/output/ijcnn1.png`
- `examples/option1_quick_proof/output/ijcnn1_confusion.png` (baseline confusion matrix diagnostics)

- `examples/comparison.png` (global aggregate timing figure)
