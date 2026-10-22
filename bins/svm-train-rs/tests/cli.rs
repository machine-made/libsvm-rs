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
    Path::new(env!("CARGO_BIN_EXE_svm-train-rs"))
}

#[test]
fn no_args_prints_help_and_exits_nonzero() {
    let output = Command::new(bin_path()).output().unwrap();
    assert!(!output.status.success());

    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("Usage: svm-train"));
}

#[test]
fn cross_validation_requires_at_least_two_folds() {
    let output = Command::new(bin_path())
        .arg("-v")
        .arg("1")
        .arg(data_file("heart_scale"))
        .output()
        .unwrap();
    assert!(!output.status.success());

    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("n-fold cross validation: n must >= 2"));
}

#[test]
fn training_writes_model_file() {
    let dir = unique_tmp_dir("svm-train");
    let model_path = dir.join("heart.model");

    let output = Command::new(bin_path())
        .arg("-q")
        .arg(data_file("heart_scale"))
        .arg(&model_path)
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8(output.stderr).unwrap()
    );
    assert!(model_path.exists());

    let content = fs::read_to_string(&model_path).unwrap();
    assert!(content.contains("svm_type"));
    assert!(content.contains("SV"));
}

#[test]
fn cross_validation_prints_accuracy() {
    let output = Command::new(bin_path())
        .arg("-q")
        .arg("-v")
        .arg("3")
        .arg(data_file("heart_scale"))
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8(output.stderr).unwrap()
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("Cross Validation Accuracy"));
}

#[test]
fn cross_validation_accepts_flag_order_variants() {
    let output = Command::new(bin_path())
        .arg("-v")
        .arg("3")
        .arg("-q")
        .arg(data_file("heart_scale"))
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8(output.stderr).unwrap()
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("Cross Validation Accuracy"));
}

#[test]
fn cross_validation_accepts_permuted_flag_chunk_order() {
    let mut orders = Vec::new();
    orders.push(vec!["-q", "-v", "3"]);
    orders.push(vec!["-v", "3", "-q"]);
    orders.push(vec!["-v", "3", "-q", "-h", "1"]);
    orders.push(vec!["-h", "1", "-v", "3", "-q"]);

    for flag_order in orders.iter() {
        let output = Command::new(bin_path())
            .args(flag_order.iter().copied())
            .arg(data_file("heart_scale"))
            .output()
            .unwrap();

        assert!(
            output.status.success(),
            "flags {:?} failed, stderr: {}",
            flag_order,
            String::from_utf8(output.stderr).unwrap()
        );
        let stdout = String::from_utf8(output.stdout).unwrap();
        assert!(stdout.contains("Cross Validation Accuracy"));
    }
}

#[test]
fn cross_validation_missing_fold_count_uses_help() {
    let output = Command::new(bin_path())
        .arg("-v")
        .arg(data_file("heart_scale"))
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("Usage: svm-train"));
}

#[test]
fn bare_dash_prints_help_and_does_not_panic() {
    // A lone "-" has no flag letter; must not panic (was: as_bytes()[1] OOB).
    let output = Command::new(bin_path())
        .arg("-")
        .arg("/tmp/x")
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("Usage: svm-train"),
        "expected usage, got: {stdout}"
    );
}

#[test]
fn random_cross_validation_flag_permutations() {
    let mut state = 0xA5A5_5A5A_FEED_F00Du64;
    let base_chunks: Vec<Vec<&str>> = vec![vec!["-q"], vec!["-v", "3"], vec!["-h", "1"]];

    for i in 0..10 {
        let mut chunks = base_chunks.clone();
        shuffle_flag_chunks(&mut chunks[..], &mut state);

        let mut command = Command::new(bin_path());
        for chunk in chunks.iter() {
            command.args(chunk.iter().copied());
        }
        let output = command.arg(data_file("heart_scale")).output().unwrap();

        assert!(
            output.status.success(),
            "iteration {i}, seed {state}, stderr: {}",
            String::from_utf8(output.stderr).unwrap()
        );
        let stdout = String::from_utf8(output.stdout).unwrap();
        assert!(stdout.contains("Cross Validation Accuracy"));
    }
}
