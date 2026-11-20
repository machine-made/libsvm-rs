# libsvm-rs Fuzz Targets

These targets exercise the text loaders with arbitrary bytes:

- `parse_problem` calls `load_problem_from_reader`.
- `parse_model` calls `load_model_from_reader`.

The pinned fuzz toolchain is recorded at the repository root in
`rust-toolchain-fuzz.toml`.

## Heavy checks

Run `bash scripts/heavy_x3.sh fuzz` from the repository root on x3 to replay both saved corpora and fuzz each parser.
Set `FUZZ_SECONDS` to change the default 300 seconds; the script removes its nightly and cargo-fuzz even after failures.
Run `bash scripts/heavy_x3.sh bench` from the repository root on x3 to measure Rust against C.
Both commands save logs under `heavy-logs/`; add `--dry-run` to preview the commands.

Fuzzing runs only when someone starts it.
The Kani job runs on pull requests that change `crates/libsvm/src`.
Crash files are copied from `crates/libsvm/fuzz/artifacts/` into `heavy-logs/`.
Before tagging a security release, set `FUZZ_SECONDS=900` to fuzz each parser for at least 15 minutes.
