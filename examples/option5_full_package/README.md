# Option 5: Full Paper-Style Package

This folder runs all datasets from the plan:
- `ijcnn1`, `covtype_scale`, `yearpredictionmsd`, `higgs`, `rcv1_binary`

## Run

```bash
bash examples/option5_full_package/run.sh
```

`run.sh` downloads Option 5 datasets and runs the full Python benchmark/plot pipeline.

## Outputs

- `examples/option5_full_package/output/results.json`
- `examples/option5_full_package/output/report.md`
- `examples/option5_full_package/output/scientific_demo.npz`
- Dataset figures under `examples/option5_full_package/output/*.png`
- Baseline diagnostics under `examples/option5_full_package/output/*_confusion.png` (classification) and `*_residuals.png` (regression)

## Note

This is the most expensive option in disk, time, and compute.

- `examples/comparison.png` (global aggregate timing figure)
