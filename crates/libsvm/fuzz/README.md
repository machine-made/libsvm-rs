# libsvm-rs Fuzz Targets

These targets exercise the text loaders with arbitrary bytes:

- `parse_problem` calls `load_problem_from_reader`.
- `parse_model` calls `load_model_from_reader`.

The pinned fuzz toolchain is recorded at the repository root in
`rust-toolchain-fuzz.toml`.

## Local Quick Check

```sh
cargo install cargo-fuzz --locked
cd crates/libsvm
cargo +nightly-2026-03-08 fuzz run parse_problem -- -max_total_time=60
cargo +nightly-2026-03-08 fuzz run parse_model -- -max_total_time=60
```

Before tagging a security release, run each target for at least 15 minutes:

```sh
cargo +nightly-2026-03-08 fuzz run parse_problem -- -max_total_time=900
cargo +nightly-2026-03-08 fuzz run parse_model -- -max_total_time=900
```

Crash artifacts are written under `fuzz/artifacts/<target>/`.
