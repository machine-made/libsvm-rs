# Examples

This directory now contains both developer-oriented Rust examples and scientific benchmark demos.

## Structure

- `examples/basics/` — minimal beginner examples
- `examples/api/` — focused API workflows (persistence, CV grid search, iris)
- `examples/scientific/` — Rust-vs-C++ benchmark/demo options
- `examples/integrations/` — integration examples (Axum server, wasm inference)
- `examples/common/` — shared Python tooling for scientific demos

## Task runner

Use the examples Makefile to simplify common workflows:

```bash
make -C examples help
make -C examples option1
make -C examples bench-significant
make -C examples comparison
```

## Rust examples (`cargo run --example ...`)

Run from repository root:

```bash
cargo run -p libsvm-rs --example minimal_classification
cargo run -p libsvm-rs --example model_persistence
cargo run -p libsvm-rs --example cross_validation_grid_search
cargo run -p libsvm-rs --example iris_classification
```

## Scientific options

Each option folder includes `README.md`, `config.json`, and `run.sh`:

1. `examples/option1_quick_proof/`
2. `examples/option2_balanced_science/`
3. `examples/option3_large_scale_headline/`
4. `examples/option4_sparse_stress/`
5. `examples/option5_full_package/`

Run one option end-to-end:

```bash
bash examples/option1_quick_proof/run.sh
```

## Outputs

- Per-option outputs: `examples/option*/output/`
- Per-dataset baseline diagnostics: `*_confusion.png` (classification), `*_residuals.png` (regression)
- Global timing figure: `examples/comparison.png`
- Global timing summary: `examples/comparison_summary.json`

## Notes

- Option 3 and Option 5 include `higgs` and are computationally heavy.
- The global comparison figure is regenerated after each scientific option run.
