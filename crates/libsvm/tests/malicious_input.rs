//! Regression tests for adversarial / malformed input files.
//!
//! Each fixture under `tests/malicious/` exercises one specific attack
//! vector against the problem or model loader. The test names document
//! the vector, and the assertions lock in the loader's rejection path
//! and error message so future refactors cannot silently accept any of
//! these inputs.
//!
//! Fixtures are kept minimal (bytes, not kilobytes) so the attack
//! corpus stays reviewable alongside the code it guards.

use std::path::PathBuf;

use libsvm_rs::io::{
    load_model, load_model_from_reader_with_options, load_problem,
    load_problem_from_reader_with_options, LoadOptions,
};

#[cfg(feature = "serde")]
use libsvm_rs::{KernelType, SvmModel, SvmNode, SvmParameter};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("malicious")
        .join(name)
}

fn assert_err_contains<E: std::fmt::Display, T: std::fmt::Debug>(
    result: Result<T, E>,
    needle: &str,
) {
    match result {
        Ok(v) => panic!(
            "expected loader error containing {:?}, got successful parse: {:?}",
            needle, v
        ),
        Err(e) => {
            let msg = format!("{}", e);
            assert!(
                msg.contains(needle),
                "expected error to contain {:?}, got: {}",
                needle,
                msg
            );
        }
    }
}

#[cfg(feature = "serde")]
fn valid_serde_model() -> SvmModel {
    SvmModel {
        param: SvmParameter {
            kernel_type: KernelType::Rbf,
            gamma: 0.5,
            ..SvmParameter::default()
        },
        nr_class: 2,
        sv: vec![vec![SvmNode {
            index: 1,
            value: 0.5,
        }]],
        sv_coef: vec![vec![0.25]],
        rho: vec![0.0],
        prob_a: Vec::new(),
        prob_b: Vec::new(),
        prob_density_marks: Vec::new(),
        sv_indices: Vec::new(),
        label: vec![1, -1],
        n_sv: vec![1, 0],
    }
}

#[cfg(feature = "serde")]
fn assert_serde_model_err_contains(model: SvmModel, needle: &str) {
    let json = serde_json::to_string(&model).unwrap();
    assert_err_contains(serde_json::from_str::<SvmModel>(&json), needle);
}

// ─── Model-file vectors ──────────────────────────────────────────────

#[test]
fn rejects_huge_total_sv_preallocation_vector() {
    // A tiny file claiming total_sv = 999_999_999 would previously have
    // triggered a ~TB-scale preallocation before reading any data. The
    // loader now rejects the header value outright.
    assert_err_contains(
        load_model(&fixture("huge_total_sv.model")),
        "total_sv exceeds limit",
    );
}

#[test]
fn rejects_huge_nr_class() {
    // nr_class > MAX_NR_CLASS caps O(k²) downstream work (multiclass
    // probability estimation, group_classes) from a single header field.
    assert_err_contains(
        load_model(&fixture("huge_nr_class.model")),
        "nr_class exceeds limit",
    );
}

#[test]
fn rejects_mismatched_n_sv_sum() {
    // sum(n_sv) != total_sv means either a corrupted file or an attempt
    // to desync downstream assumptions. validate_model_header catches it
    // before any SV allocation.
    assert_err_contains(
        load_model(&fixture("mismatched_n_sv_sum.model")),
        "sum of nr_sv entries",
    );
}

#[test]
fn rejects_rho_length_mismatch() {
    // Classification with nr_class = 3 must have rho.len() = 3; a single
    // rho value implies a structural inconsistency.
    assert_err_contains(
        load_model(&fixture("rho_length_mismatch.model")),
        "rho has 1 entries, expected 3",
    );
}

#[test]
fn rejects_label_on_regression() {
    // `label` is classification-only; its presence on an SVR file is a
    // category confusion that could smuggle unexpected data into
    // downstream consumers.
    assert_err_contains(
        load_model(&fixture("label_on_regression.model")),
        "label is only valid for classification",
    );
}

#[test]
fn rejects_prob_density_marks_on_csvc() {
    // prob_density_marks is meaningful only for one-class SVM; carrying
    // it on c_svc is a similar category confusion.
    assert_err_contains(
        load_model(&fixture("prob_density_marks_on_csvc.model")),
        "prob_density_marks is only valid for one-class SVM",
    );
}

#[test]
fn rejects_sv_feature_indices_not_ascending() {
    // SV feature indices must be ascending (same invariant the problem
    // parser enforces). Out-of-order indices can confuse any downstream
    // kernel routine that relies on sortedness.
    assert_err_contains(
        load_model(&fixture("sv_feature_indices_not_ascending.model")),
        "feature indices must be ascending",
    );
}

#[cfg(feature = "serde")]
#[test]
fn serde_rejects_rho_length_mismatch() {
    let mut model = valid_serde_model();
    model.rho.clear();
    assert_serde_model_err_contains(model, "rho has 0 entries, expected 1");
}

#[cfg(feature = "serde")]
#[test]
fn serde_rejects_label_length_mismatch() {
    let mut model = valid_serde_model();
    model.label.pop();
    assert_serde_model_err_contains(model, "label has 1 entries");
}

#[cfg(feature = "serde")]
#[test]
fn serde_rejects_sv_coef_length_mismatch() {
    let mut model = valid_serde_model();
    model.sv_coef[0].clear();
    assert_serde_model_err_contains(model, "sv_coef row 0 has 0 entries, expected 1");
}

#[cfg(feature = "serde")]
#[test]
fn serde_rejects_non_finite_gamma() {
    let mut model = valid_serde_model();
    model.param.gamma = f64::INFINITY;
    assert_serde_model_err_contains(model, "expected f64");
}

#[cfg(feature = "serde")]
#[test]
fn serde_rejects_non_finite_rho() {
    let mut model = valid_serde_model();
    model.rho[0] = f64::NAN;
    assert_serde_model_err_contains(model, "expected f64");
}

#[cfg(feature = "serde")]
#[test]
fn serde_rejects_non_finite_sv_coef() {
    let mut model = valid_serde_model();
    model.sv_coef[0][0] = f64::INFINITY;
    assert_serde_model_err_contains(model, "expected f64");
}

#[cfg(feature = "serde")]
#[test]
fn serde_rejects_negative_gamma() {
    let mut model = valid_serde_model();
    model.param.gamma = -1.0;
    assert_serde_model_err_contains(model, "gamma must be >= 0");
}

#[cfg(feature = "serde")]
#[test]
fn serde_rejects_negative_degree() {
    let mut model = valid_serde_model();
    model.param.kernel_type = KernelType::Polynomial;
    model.param.degree = -1;
    assert_serde_model_err_contains(model, "degree must be >= 0");
}

// ─── Problem-file vectors ────────────────────────────────────────────

#[test]
fn rejects_feature_index_out_of_range() {
    assert_err_contains(
        load_problem(&fixture("feature_index_out_of_range.libsvm")),
        "feature index 10000001 exceeds limit",
    );
}

#[test]
fn rejects_line_over_max_line_len() {
    // The fixture contains one ~695-byte line; with max_line_len=100 the
    // read helper refuses to grow the line buffer past the cap.
    let path = fixture("long_line.libsvm");
    let bytes = std::fs::read(&path).unwrap();
    let opts = LoadOptions {
        max_line_len: 100,
        ..LoadOptions::default()
    };
    assert_err_contains(
        load_problem_from_reader_with_options(bytes.as_slice(), &opts),
        "max_line_len",
    );
}

#[test]
fn rejects_file_over_max_bytes() {
    // Same fixture, now tripping the total-bytes cap instead. With a
    // 64-byte cap on a ~695-byte file the first line is enough to fail.
    let path = fixture("long_line.libsvm");
    let bytes = std::fs::read(&path).unwrap();
    let opts = LoadOptions {
        max_bytes: 64,
        ..LoadOptions::default()
    };
    assert_err_contains(
        load_problem_from_reader_with_options(bytes.as_slice(), &opts),
        "max_bytes",
    );
}

#[test]
fn rejects_nul_byte_in_problem_line() {
    // NUL has no legal use in LIBSVM text; reject without parsing the
    // rest of the line. The fixture is built inline to avoid committing
    // a binary file.
    let payload: &[u8] = b"+1 1:0.5\0\n";
    assert_err_contains(
        load_problem_from_reader_with_options(payload, &LoadOptions::default()),
        "NUL byte",
    );
}

#[test]
fn rejects_nul_byte_in_model_sv_section() {
    // Same vector, SV-section side — covers both parsers.
    let mut payload: Vec<u8> =
        b"svm_type c_svc\nkernel_type linear\nnr_class 2\ntotal_sv 1\nrho 0\nlabel 1 -1\nnr_sv 1 0\nSV\n0.1 1:0.5"
            .to_vec();
    payload.push(0);
    payload.push(b'\n');
    assert_err_contains(
        load_model_from_reader_with_options(payload.as_slice(), &LoadOptions::default()),
        "NUL byte",
    );
}
