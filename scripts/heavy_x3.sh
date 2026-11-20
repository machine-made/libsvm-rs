#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# Run fuzzing or Rust versus C benchmarks on x3, logging results and cleaning up fuzz tools.
# Add --dry-run to print commands without installing tools or running checks.
set -euo pipefail

usage() {
    printf 'Usage: bash scripts/heavy_x3.sh [--dry-run] {fuzz|bench}\n' >&2
    exit 2
}

dry_run=false
args=()
for arg in "$@"; do
    if [[ $arg == --dry-run && $dry_run == false ]]; then
        dry_run=true
    else
        args+=("$arg")
    fi
done
[[ ${#args[@]} -eq 1 ]] || usage
mode=${args[0]}
case "$mode" in fuzz|bench) ;; *) usage ;; esac
seconds=${FUZZ_SECONDS:-300}
if [[ $mode == fuzz && ! $seconds =~ ^[1-9][0-9]*$ ]]; then
    usage
fi

repo_root=$(realpath -- "$(dirname -- "${BASH_SOURCE[0]}")/..")
stamp=$(date -u +%Y-%m-%dT%H-%M-%S)-$$
log_dir="$repo_root/heavy-logs"
log="$log_dir/$mode-$stamp.log"
fuzz_root="$repo_root/crates/libsvm/fuzz"
corpus_root="$log_dir/corpus-$stamp"
nightly=$(awk -F '"' '/^channel[[:space:]]*=/ {print $2}' "$repo_root/rust-toolchain-fuzz.toml")
if [[ $mode == fuzz && ! $nightly =~ ^nightly-[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    printf 'Expected one dated nightly in rust-toolchain-fuzz.toml.\n' >&2
    exit 1
fi

run() {
    printf '+ '
    printf '%q ' "$@"
    printf '\n'
    if [[ $dry_run == false ]]; then
        "$@"
    fi
}

# All heavy commands, including compiler children of Python, share these limits.
heavy() {
    local directory=$1
    shift
    local wrapper=()
    if command -v jobscope >/dev/null 2>&1; then
        wrapper=(jobscope)
    fi
    run env -C "$directory" RAYON_NUM_THREADS=2 CARGO_BUILD_JOBS=2 \
        nice -n 19 "${wrapper[@]}" "$@"
}

remove_nightly=false
remove_fuzz=false
targets=0
crashes=0
marker=

finish() {
    local status=$?
    trap - EXIT INT TERM HUP
    set +e
    if [[ $remove_fuzz == true ]]; then
        heavy "$repo_root" cargo uninstall cargo-fuzz || status=1
    fi
    if [[ $remove_nightly == true ]]; then
        heavy "$repo_root" rustup toolchain uninstall "$nightly" || status=1
    fi
    if [[ $mode == fuzz ]]; then
        local artifacts=()
        shopt -s nullglob globstar dotglob
        local artifact
        for artifact in "$fuzz_root"/artifacts/**/*; do
            [[ -f $artifact ]] && artifacts+=("$artifact")
        done
        crashes=${#artifacts[@]}
        if [[ $crashes -gt 0 ]]; then
            run cp -a -- "$fuzz_root/artifacts" "$log_dir/artifacts-$stamp" || status=1
            printf 'Crash artifacts: %s\n' "${artifacts[@]}"
            status=1
        fi
    fi
    if [[ -n $marker ]]; then
        rm -f -- "$marker" || status=1
    fi
    local result=FAIL
    [[ $status -eq 0 ]] && result=OK
    if [[ $mode == fuzz ]]; then
        printf 'RESULT: %s targets=%s crashes=%s seconds_per_target=%s\n' \
            "$result" "$targets" "$crashes" "$seconds"
    else
        printf 'RESULT: %s\n' "$result"
    fi
    exit "$status"
}

run_fuzz() {
    # Arm cleanup before installation, so partial installs also get removed.
    remove_nightly=true
    heavy "$repo_root" rustup toolchain install "$nightly" --profile minimal
    remove_fuzz=true
    heavy "$repo_root" cargo install cargo-fuzz --locked
    local target
    for target in parse_problem parse_model; do
        run mkdir -p -- "$corpus_root/$target"
        run cp -a -- "$fuzz_root/corpus/$target/." "$corpus_root/$target/"
    done
    # Both seed sets must pass before either timed run starts.
    for target in parse_problem parse_model; do
        targets=$((targets + 1))
        heavy "$fuzz_root" cargo +"$nightly" fuzz run "$target" "$corpus_root/$target" \
            -- -runs=0 -rss_limit_mb=2048
    done
    local status=0
    for target in parse_problem parse_model; do
        if heavy "$fuzz_root" cargo +"$nightly" fuzz run "$target" "$corpus_root/$target" \
            -- -max_total_time="$seconds" -rss_limit_mb=2048; then
            :
        else
            printf 'Fuzz target %s failed.\n' "$target"
            status=1
        fi
    done
    return "$status"
}

run_bench() {
    if [[ $dry_run == false ]]; then
        marker=$(mktemp "$log_dir/bench-start-XXXXXX")
    fi
    heavy "$repo_root" python3 scripts/benchmark_compare.py
    if [[ $dry_run == true ]]; then
        printf 'Require both reference/benchmark_report.md and reference/benchmark_results.json newer than the run marker.\n'
    else
        local report
        for report in benchmark_report.md benchmark_results.json; do
            if [[ ! -f "$repo_root/reference/$report" || ! "$repo_root/reference/$report" -nt $marker ]]; then
                printf 'Benchmark did not rewrite reference/%s.\n' "$report" >&2
                return 1
            fi
        done
    fi
    run cat "$repo_root/reference/benchmark_report.md"
}

printf 'Log: %s\n' "$log"
if [[ $dry_run == true ]]; then
    "run_$mode"
    if [[ $mode == fuzz ]]; then
        heavy "$repo_root" cargo uninstall cargo-fuzz
        heavy "$repo_root" rustup toolchain uninstall "$nightly"
        printf 'Copy any crash artifacts to heavy-logs/artifacts-%s; require both targets to pass and crashes=0.\n' "$stamp"
    fi
    printf 'Dry run: no commands executed and no RESULT recorded.\n'
else
    mkdir -p -- "$log_dir"
    (
        trap finish EXIT
        trap 'exit 130' INT
        trap 'exit 143' TERM
        trap 'exit 129' HUP
        "run_$mode"
    ) 2>&1 | tee "$log"
fi
