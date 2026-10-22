#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python3 -u "$ROOT/examples/common/download_reference_data.py" --datasets option2
python3 -u "$ROOT/examples/common/run_scientific_demo.py" --config "$ROOT/examples/option2_balanced_science/config.json" "$@"
