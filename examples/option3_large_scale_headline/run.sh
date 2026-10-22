#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python3 -u "$ROOT/examples/common/download_reference_data.py" --datasets option3
python3 -u "$ROOT/examples/common/run_scientific_demo.py" --config "$ROOT/examples/option3_large_scale_headline/config.json" "$@"
