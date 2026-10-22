# Option 2: Balanced Scientific Demo

This folder runs the recommended balanced demo on:
- `ijcnn1` (binary classification)
- `covtype_scale` (multiclass classification)
- `yearpredictionmsd` (regression)

## Run

```bash
bash examples/option2_balanced_science/run.sh
```

`run.sh` downloads Option 2 datasets and runs the Python pipeline to generate figures.

## Outputs

- `examples/option2_balanced_science/output/results.json`
- `examples/option2_balanced_science/output/report.md`
- `examples/option2_balanced_science/output/scientific_demo.npz`
- `examples/option2_balanced_science/output/ijcnn1.png`
- `examples/option2_balanced_science/output/covtype_scale.png`
- `examples/option2_balanced_science/output/yearpredictionmsd.png`
- `examples/option2_balanced_science/output/*_confusion.png` (classification baseline diagnostics)
- `examples/option2_balanced_science/output/*_residuals.png` (regression baseline diagnostics)

- `examples/comparison.png` (global aggregate timing figure)
