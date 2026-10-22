# Option 4: Sparse High-Dimensional Stress Demo

This folder extends the balanced demo by adding `rcv1_binary`.

## Run

```bash
bash examples/option4_sparse_stress/run.sh
```

`run.sh` downloads Option 4 datasets and runs the Python pipeline to generate figures.

## Outputs

- `examples/option4_sparse_stress/output/results.json`
- `examples/option4_sparse_stress/output/report.md`
- `examples/option4_sparse_stress/output/scientific_demo.npz`
- Dataset figures under `examples/option4_sparse_stress/output/*.png`
- Baseline diagnostics under `examples/option4_sparse_stress/output/*_confusion.png` (classification) and `*_residuals.png` (regression)

## Note

`rcv1_binary` includes 47,236 features and a large test split.

- `examples/comparison.png` (global aggregate timing figure)
