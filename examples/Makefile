.PHONY: help option1 option2 option3 option4 option5 comparison bench-significant rust-examples minimal persistence grid iris server wasm

help:
	@echo "Scientific: option1 option2 option3 option4 option5 comparison bench-significant"
	@echo "Rust API: minimal persistence grid iris rust-examples"
	@echo "Integration: server wasm"

option1:
	bash option1_quick_proof/run.sh

option2:
	bash option2_balanced_science/run.sh

option3:
	bash option3_large_scale_headline/run.sh

option4:
	bash option4_sparse_stress/run.sh

option5:
	bash option5_full_package/run.sh

comparison:
	python3 common/make_comparison_figure.py --root .. --out examples/comparison.png --summary examples/comparison_summary.json --min-runs 3

bench-significant:
	cd .. && BENCH_WARMUP=3 BENCH_RUNS=20 python3 scripts/benchmark_compare.py
	python3 common/make_comparison_figure.py --root .. --out examples/comparison.png --summary examples/comparison_summary.json --min-runs 3

rust-examples:
	cd .. && cargo run -p libsvm-rs --example minimal_classification
	cd .. && cargo run -p libsvm-rs --example model_persistence
	cd .. && cargo run -p libsvm-rs --example cross_validation_grid_search
	cd .. && cargo run -p libsvm-rs --example iris_classification

minimal:
	cd .. && cargo run -p libsvm-rs --example minimal_classification

persistence:
	cd .. && cargo run -p libsvm-rs --example model_persistence

grid:
	cd .. && cargo run -p libsvm-rs --example cross_validation_grid_search

iris:
	cd .. && cargo run -p libsvm-rs --example iris_classification

server:
	bash integrations/prediction_server/run.sh

wasm:
	bash integrations/wasm_inference/run.sh
