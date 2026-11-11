# SPDX-License-Identifier: BSD-3-Clause

# Contributing

## Pull requests

Send all changes as pull requests. Do not push to `main`.

## Gates

The list below is the CI gate list. Each pull request must pass these gates.
Run the same commands on your machine before you open the pull request.

```
# CI MSRV job
cargo +1.85.0 check --locked --all-features

cargo fmt --all -- --check
cargo clippy --locked --all-targets --all-features -- -D warnings

# Build C reference tools
make -C vendor/libsvm

cargo test --locked --all-features
cargo test --locked --no-default-features
cargo test --locked --features rayon
cargo test --doc --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features

cargo build --release --locked -p svm-train-rs -p svm-predict-rs
bash scripts/compare_references.sh
```

## Port code

This project is a Rust port of LIBSVM. Before you add or change port code:

1. Freeze a reference from the original C code. The reference holds the
   inputs and the outputs of the original.
2. Show that the Rust code gives the same outputs as the original, with the
   tolerance stated for each output. Build the release tools, then run the
   parity script.

   ```
   cargo build --release --locked -p svm-train-rs -p svm-predict-rs
   bash scripts/compare_references.sh
   ```

A pull request that has no frozen reference and no parity result is not
accepted.

## Licence

Your contribution is licensed under BSD-3-Clause, the licence of this project.
