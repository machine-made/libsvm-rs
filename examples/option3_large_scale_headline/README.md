# Option 3: Large-Scale Headline Demo

This folder extends the balanced demo by adding `higgs`.

## Run

```bash
bash examples/option3_large_scale_headline/run.sh
```

`run.sh` downloads Option 3 datasets (including large `higgs`) and runs the Python pipeline.

## Outputs

- `examples/option3_large_scale_headline/output/results.json`
- `examples/option3_large_scale_headline/output/report.md`
- `examples/option3_large_scale_headline/output/scientific_demo.npz`
- Dataset figures under `examples/option3_large_scale_headline/output/*.png`
- Baseline diagnostics under `examples/option3_large_scale_headline/output/*_confusion.png` (classification) and `*_residuals.png` (regression)

## Note

`higgs` download and execution are heavy by design.

- `examples/comparison.png` (global aggregate timing figure)
