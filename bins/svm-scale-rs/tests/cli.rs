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
    Path::new(env!("CARGO_BIN_EXE_svm-scale-rs"))
}

#[test]
fn no_args_prints_help_and_exits_nonzero() {
    let output = Command::new(bin_path()).output().unwrap();
    assert!(!output.status.success());

    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("Usage: svm-scale"));
}

#[test]
fn inconsistent_bounds_return_error() {
    let output = Command::new(bin_path())
        .arg("-l")
        .arg("1")
        .arg("-u")
        .arg("0")
        .arg(data_file("heart_scale"))
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("inconsistent lower/upper specification"));
}

#[test]
fn save_then_restore_produces_same_scaled_output() {
    let dir = unique_tmp_dir("svm-scale");
    let data_path = dir.join("tiny.scale");
    let scale_path = dir.join("tiny.params");

    fs::write(&data_path, "1 1:1.0 3:2.0\n-1 1:0.0 2:4.0 3:1.0\n1 2:2.0\n").unwrap();

    let out_save = Command::new(bin_path())
        .arg("-s")
        .arg(&scale_path)
        .arg(&data_path)
        .output()
        .unwrap();
    assert!(
        out_save.status.success(),
        "stderr: {}",
        String::from_utf8(out_save.stderr).unwrap()
    );
    assert!(scale_path.exists());

    let out_restore = Command::new(bin_path())
        .arg("-r")
        .arg(&scale_path)
        .arg(&data_path)
        .output()
        .unwrap();
    assert!(
        out_restore.status.success(),
        "stderr: {}",
        String::from_utf8(out_restore.stderr).unwrap()
    );

    let save_stdout = String::from_utf8(out_save.stdout).unwrap();
    let restore_stdout = String::from_utf8(out_restore.stdout).unwrap();
    assert_eq!(save_stdout, restore_stdout);
}

#[test]
fn scaling_emits_non_empty_output() {
    let output = Command::new(bin_path())
        .arg(data_file("heart_scale"))
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8(output.stderr).unwrap()
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(!stdout.trim().is_empty());
}

#[test]
fn negative_feature_index_returns_error_instead_of_panicking() {
    let dir = unique_tmp_dir("svm-scale-neg-index");
    let data_path = dir.join("negative_index.scale");
    fs::write(&data_path, "1 -1:0.5 1:1.0\n").unwrap();

    let output = Command::new(bin_path()).arg(&data_path).output().unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains("feature index -1 out of valid range [1, 10000000]"),
        "stderr: {stderr}"
    );
}

#[test]
fn y_scaling_requires_two_bounds() {
    let dir = unique_tmp_dir("svm-scale-bad-y");
    let data_path = dir.join("tiny.scale");
    fs::write(&data_path, "1 1:1.0\n").unwrap();

    let output = Command::new(bin_path())
        .arg("-y")
        .arg("0.0")
        .arg(&data_path)
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("Usage: svm-scale"));
}

#[test]
fn scaling_with_mixed_flag_order_is_stable() {
    let dir = unique_tmp_dir("svm-scale-order");
    let output_path = dir.join("tiny.params");
    let data_path = dir.join("tiny.scale");
    fs::write(&data_path, "1 1:2.0 2:4.0\n-1 2:1.0\n").unwrap();

    let output = Command::new(bin_path())
        .arg("-u")
        .arg("1")
        .arg("-l")
        .arg("-1")
        .arg("-s")
        .arg(&output_path)
        .arg(&data_path)
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8(output.stderr).unwrap()
    );
    assert!(output_path.exists());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(!stdout.trim().is_empty());
}

#[test]
fn scaling_accepts_permuted_scale_flags() {
    let dir = unique_tmp_dir("svm-scale-permuted-flags");
    let data_path = dir.join("tiny.scale");
    fs::write(&data_path, "1 1:2.0 2:4.0\n-1 2:1.0\n").unwrap();

    let orders: Vec<Vec<&str>> = vec![
        vec!["-u", "1", "-l", "-1", "-s"],
        vec!["-l", "-1", "-u", "1", "-s"],
        vec!["-s", "-u", "1", "-l", "-1"],
        vec!["-u", "1", "-s", "-l", "-1"],
    ];

    for (i, order) in orders.iter().enumerate() {
        let scale_path = dir.join(format!("tiny-{i}.params"));
        let mut command = Command::new(bin_path());
        for token in order.iter() {
            if *token == "-s" {
                command.arg(token).arg(&scale_path);
            } else {
                command.arg(token);
            }
        }
        let output = command.arg(&data_path).output().unwrap();

        assert!(
            output.status.success(),
            "order {:?} failed, stderr: {}",
            order,
            String::from_utf8(output.stderr).unwrap()
        );
        assert!(scale_path.exists());
    }
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
        stdout.contains("Usage: svm-scale"),
        "expected usage, got: {stdout}"
    );
}

#[test]
fn random_scale_flag_permutations_still_generate_output() {
    let dir = unique_tmp_dir("svm-scale-fuzz");
    let data_path = dir.join("tiny.scale");
    fs::write(&data_path, "1 1:2.0 2:4.0\n-1 2:1.0\n").unwrap();

    let base_chunks: Vec<Vec<&str>> = vec![vec!["-u", "1"], vec!["-l", "-1"], vec!["-s", ""]];

    let mut state = 0xC0FF_EE00_5A5A_1234u64;
    for i in 0..10 {
        let mut chunks = base_chunks.clone();
        shuffle_flag_chunks(&mut chunks[..], &mut state);

        let output_path = dir.join(format!("tiny-fuzz-{i}.params"));
        let mut command = Command::new(bin_path());
        for chunk in chunks.iter() {
            if chunk[0] == "-s" {
                command.arg("-s").arg(&output_path);
            } else {
                command.args(chunk.iter().copied());
            }
        }
        let output = command.arg(&data_path).output().unwrap();

        assert!(
            output.status.success(),
            "iteration {i}, state {state}, stderr: {}",
            String::from_utf8(output.stderr).unwrap()
        );
        assert!(output_path.exists());
        assert!(!String::from_utf8(output.stdout).unwrap().is_empty());
    }
}
