#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXAMPLE_DIR="$ROOT/examples/integrations/wasm_inference"
OUT_DIR="$EXAMPLE_DIR/output"
TMP_DIR="$OUT_DIR/tmp"
PKG_DIR="$EXAMPLE_DIR/pkg"
WASM_CRATE="$EXAMPLE_DIR/wasm_module/Cargo.toml"

WARMUP="${WASM_WARMUP:-3}"
RUNS="${WASM_RUNS:-20}"
TRAIN_ROWS="${WASM_TRAIN_ROWS:-180}"
CXX_BIN="${CXX:-c++}"

mkdir -p "$OUT_DIR" "$TMP_DIR" "$PKG_DIR"

if ! command -v node >/dev/null 2>&1; then
  echo "node is required" >&2
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required" >&2
  exit 1
fi
if ! command -v "$CXX_BIN" >/dev/null 2>&1; then
  echo "C++ compiler '$CXX_BIN' is required" >&2
  exit 1
fi

rustup target add wasm32-unknown-unknown >/dev/null

if ! command -v wasm-bindgen >/dev/null 2>&1; then
  echo "Installing wasm-bindgen-cli (one-time)..."
  cargo install wasm-bindgen-cli --locked >/dev/null
fi

echo "Building wasm module..."
cargo build --manifest-path "$WASM_CRATE" --release --target wasm32-unknown-unknown >/dev/null

WASM_BIN="$EXAMPLE_DIR/wasm_module/target/wasm32-unknown-unknown/release/libsvm_wasm_inference.wasm"
wasm-bindgen --target nodejs --out-dir "$PKG_DIR" "$WASM_BIN"

TRAIN_FILE="$TMP_DIR/heart_scale.train.svm"
TEST_FILE="$TMP_DIR/heart_scale.test.svm"
python3 - "$ROOT/data/heart_scale" "$TRAIN_ROWS" "$TRAIN_FILE" "$TEST_FILE" <<'PY'
import sys
from pathlib import Path

src = Path(sys.argv[1])
train_rows = int(sys.argv[2])
train_out = Path(sys.argv[3])
test_out = Path(sys.argv[4])

lines = src.read_text(encoding="utf-8").splitlines()
if train_rows <= 0 or train_rows >= len(lines):
    raise SystemExit(f"invalid TRAIN_ROWS={train_rows}, total lines={len(lines)}")

train_out.write_text("\n".join(lines[:train_rows]) + "\n", encoding="utf-8")
test_out.write_text("\n".join(lines[train_rows:]) + "\n", encoding="utf-8")
print(f"Prepared split: train={train_rows}, test={len(lines) - train_rows}")
PY

WASM_JSON="$OUT_DIR/wasm_node_raw.json"
node "$EXAMPLE_DIR/benchmark_node.cjs" \
  "$PKG_DIR/libsvm_wasm_inference.js" \
  "$TRAIN_FILE" \
  "$TEST_FILE" \
  "$WARMUP" \
  "$RUNS" >"$WASM_JSON"

CPP_BIN="$OUT_DIR/cpp_inprocess_bench"
echo "Building C++ in-process benchmark harness..."
"$CXX_BIN" -O3 -std=c++17 \
  -I"$ROOT/vendor/libsvm" \
  "$EXAMPLE_DIR/cpp_inprocess_bench.cpp" \
  "$ROOT/vendor/libsvm/svm.cpp" \
  -o "$CPP_BIN"

CPP_JSON="$OUT_DIR/cpp_raw.json"
"$CPP_BIN" "$TRAIN_FILE" "$TEST_FILE" "$WARMUP" "$RUNS" >"$CPP_JSON"

python3 "$EXAMPLE_DIR/bench_wasm_vs_cpp.py" \
  --test "$TEST_FILE" \
  --wasm-json "$WASM_JSON" \
  --cpp-json "$CPP_JSON" \
  --out-dir "$OUT_DIR" \
  --warmup "$WARMUP" \
  --runs "$RUNS"

python3 "$ROOT/examples/common/make_comparison_figure.py" \
  --root "$ROOT" \
  --out examples/comparison.png \
  --summary examples/comparison_summary.json \
  --min-runs 3

echo "Wrote $OUT_DIR/results.json"
echo "Wrote $OUT_DIR/report.md"
echo "Wrote $OUT_DIR/wasm_vs_cpp.png"
echo "Updated $ROOT/examples/comparison.png"
