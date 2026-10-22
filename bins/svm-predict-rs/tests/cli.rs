use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

#[path = "../../../tests/cli_flag_helpers.rs"]
mod cli_flag_helpers;

use cli_flag_helpers::shuffle_flag_chunks;

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

fn data_file(name: &str) -> PathBuf {
    workspace_root().join("data").join(name)
}

fn unique_tmp_dir(tag: &str) -> PathBuf {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let dir = workspace_root()
        .join(".tmp")
        .join("cli-tests")
        .join(format!("{}-{}-{}", tag, std::process::id(), id));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn bin_path() -> &'static Path {
    Path::new(env!("CARGO_BIN_EXE_svm-predict-rs"))
}

#[test]
fn no_args_prints_help_and_exits_nonzero() {
    let output = Command::new(bin_path()).output().unwrap();
    assert!(!output.status.success());

    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("Usage: svm-predict"));
}

#[test]
fn prediction_writes_output_file() {
    let dir = unique_tmp_dir("svm-predict");
    let output_path = dir.join("predictions.txt");

    let output = Command::new(bin_path())
        .arg("-q")
        .arg(data_file("heart_scale"))
        .arg(data_file("heart_scale.model"))
        .arg(&output_path)
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8(output.stderr).unwrap()
    );
    assert!(output_path.exists());

    let lines = fs::read_to_string(&output_path).unwrap();
    assert_eq!(lines.lines().count(), 270);
}

#[test]
fn probability_flag_rejected_for_non_probability_model() {
    let dir = unique_tmp_dir("svm-predict-prob");
    let output_path = dir.join("predictions.txt");

    let output = Command::new(bin_path())
        .arg("-b")
        .arg("1")
        .arg(data_file("heart_scale"))
        .arg(data_file("heart_scale.model"))
        .arg(&output_path)
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("Model does not support probability estimates"));
}

#[test]
fn mixed_flag_order_preserves_quiet_prediction() {
    let dir = unique_tmp_dir("svm-predict-order");
    let output_path = dir.join("predictions.txt");

    let output = Command::new(bin_path())
        .arg("-b")
        .arg("0")
        .arg("-q")
        .arg(data_file("heart_scale"))
        .arg(data_file("heart_scale.model"))
        .arg(&output_path)
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8(output.stderr).unwrap()
    );
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(!stderr.contains("Accuracy ="));
    assert!(output_path.exists());
}

#[test]
fn quiet_flag_variants_with_probability_off() {
    let dir = unique_tmp_dir("svm-predict-permutations");
    let args = vec![vec!["-b", "0", "-q"], vec!["-q", "-b", "0"], vec!["-q"]];

    for flag_order in args {
        let output_path = dir.join(format!("predictions-{}.txt", flag_order.join("-")));
        let mut command = Command::new(bin_path());
        for flag in &flag_order {
            command.arg(flag);
        }
        let output = command
            .arg(data_file("heart_scale"))
            .arg(data_file("heart_scale.model"))
            .arg(&output_path)
            .output()
            .unwrap();

        assert!(
            output.status.success(),
            "args {:?} failed, stderr: {}",
            flag_order,
            String::from_utf8(output.stderr).unwrap()
        );
        assert!(output_path.exists());
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(!stderr.contains("Accuracy ="));
    }
}

#[test]
fn quiet_mode_suppresses_accuracy_summary() {
    let dir = unique_tmp_dir("svm-predict-quiet");
    let output_path = dir.join("predictions.txt");

    let output = Command::new(bin_path())
        .arg("-q")
        .arg(data_file("heart_scale"))
        .arg(data_file("heart_scale.model"))
        .arg(&output_path)
        .output()
        .unwrap();

    assert!(output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(!stderr.contains("Accuracy ="));
}

#[test]
fn probability_flag_without_value_prints_help() {
    let dir = unique_tmp_dir("svm-predict-bad-flag");
    let output_path = dir.join("predictions.txt");

    let output = Command::new(bin_path())
        .arg("-b")
        .arg(data_file("heart_scale"))
        .arg(data_file("heart_scale.model"))
        .arg(&output_path)
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("Usage: svm-predict"));
}

#[test]
fn bare_dash_prints_help_and_does_not_panic() {
    // A lone "-" has no flag letter; must not panic (was: as_bytes()[1] OOB).
    let output = Command::new(bin_path())
        .arg("-")
        .arg("a")
        .arg("b")
        .arg("c")
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("Usage: svm-predict"),
        "expected usage, got: {stdout}"
    );
}

#[test]
fn random_flag_chunk_permutations_preserve_quiet_behavior() {
    let dir = unique_tmp_dir("svm-predict-fuzz");
    let mut state = 0x5A5A_5A5A_DEAD_BEEFu64;
    let base_chunks: Vec<Vec<&str>> = vec![vec!["-b", "0"], vec!["-q"]];

    for i in 0..12 {
        let mut chunks = base_chunks.clone();
        shuffle_flag_chunks(&mut chunks[..], &mut state);

        let output_path = dir.join(format!("predictions-{i}.txt"));
        let mut command = Command::new(bin_path());
        for chunk in chunks.iter() {
            command.args(chunk.iter().copied());
        }
        let output = command
            .arg(data_file("heart_scale"))
            .arg(data_file("heart_scale.model"))
            .arg(&output_path)
            .output()
            .unwrap();
        let stderr = String::from_utf8(output.stderr).unwrap();

        assert!(
            output.status.success(),
            "iteration {i}, seed state {state}, stderr: {}",
            stderr
        );
        assert!(output_path.exists());
        assert!(stderr.is_empty() || !stderr.contains("Accuracy ="));
    }
}
