# Reproducing Release Artifacts

This procedure checks that a GitHub Release artifact is structurally
reproducible from the tagged source tree.

## Inputs

- Tagged source checkout, for example `v0.8.0`.
- The matching GitHub Release archive for one target.
- Rust toolchain pinned by `rust-toolchain.toml`.
- Cargo lockfile from the tag.

## Procedure

```sh
git fetch --tags
git checkout v0.8.0
cargo build --release --locked --target x86_64-unknown-linux-gnu
mkdir -p .tmp/reproduce-release
cp target/x86_64-unknown-linux-gnu/release/svm-train-rs .tmp/reproduce-release/
cp target/x86_64-unknown-linux-gnu/release/svm-predict-rs .tmp/reproduce-release/
cp target/x86_64-unknown-linux-gnu/release/svm-scale-rs .tmp/reproduce-release/
tar czf .tmp/libsvm-rs-v0.8.0-x86_64-unknown-linux-gnu.tar.gz -C .tmp/reproduce-release .
```

Then unpack the downloaded release archive and compare file names, executable
presence, and command behavior:

```sh
mkdir -p .tmp/release-downloaded .tmp/release-local
tar xzf libsvm-rs-v0.8.0-x86_64-unknown-linux-gnu.tar.gz -C .tmp/release-downloaded
tar xzf .tmp/libsvm-rs-v0.8.0-x86_64-unknown-linux-gnu.tar.gz -C .tmp/release-local
find .tmp/release-downloaded .tmp/release-local -maxdepth 1 -type f -print | sort
.tmp/release-local/svm-train-rs --help >/dev/null
.tmp/release-local/svm-predict-rs --help >/dev/null
.tmp/release-local/svm-scale-rs --help >/dev/null
```

Byte-for-byte identity is not guaranteed across every platform because archive
metadata, linker details, and platform signing/timestamp behavior can differ.
The expected result is the same executable set built from the same tag with
`--locked`, and matching CLI behavior on a small quick check.

For a stronger check, rebuild on the same operating system and target triple
used by the GitHub Release job, then compare SHA-256 hashes of the unpacked
binaries:

```sh
shasum -a 256 .tmp/release-downloaded/* .tmp/release-local/*
```
