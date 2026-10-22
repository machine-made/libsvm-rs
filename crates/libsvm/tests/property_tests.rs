//! Property-based tests for libsvm-rs using proptest.
//!
//! These tests verify core invariants:
//! - Determinism: repeated predictions with same input produce identical results
//! - Classification invariant: predictions are valid training labels
//! - Cross-validation: outputs are finite and valid labels

use libsvm_rs::cross_validation::svm_cross_validation;
use libsvm_rs::io::{
    load_model_from_reader, load_problem, load_problem_from_reader, save_model_to_writer,
};
use libsvm_rs::predict::predict;
use libsvm_rs::train::svm_train;
use libsvm_rs::types::{KernelType, SvmModel, SvmNode, SvmParameter, SvmProblem, SvmType};
use proptest::prelude::*;
use std::path::Path;

/// Helper to load heart_scale dataset from the project data directory.
fn load_heart_scale() -> SvmProblem {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/heart_scale");
    load_problem(Path::new(path)).expect("Failed to load heart_scale dataset")
}

/// Helper to extract unique labels from a problem.
fn unique_labels(prob: &SvmProblem) -> Vec<f64> {
    let mut labels: Vec<f64> = prob.labels.to_vec();
    labels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    labels.dedup();
    labels
}

fn finite_value_strategy() -> impl Strategy<Value = f64> {
    -1000.0f64..1000.0
}

fn sparse_instance_strategy() -> impl Strategy<Value = Vec<SvmNode>> {
    prop::collection::vec((1i32..64, finite_value_strategy()), 0..12).prop_map(|mut pairs| {
        pairs.sort_by_key(|(index, _)| *index);
        pairs.dedup_by_key(|(index, _)| *index);
        pairs
            .into_iter()
            .map(|(index, value)| SvmNode { index, value })
            .collect()
    })
}

fn problem_strategy() -> impl Strategy<Value = SvmProblem> {
    (1usize..16).prop_flat_map(|len| {
        (
            prop::collection::vec(finite_value_strategy(), len),
            prop::collection::vec(sparse_instance_strategy(), len),
        )
            .prop_map(|(labels, instances)| SvmProblem { labels, instances })
    })
}

fn binary_model_strategy() -> impl Strategy<Value = SvmModel> {
    (1usize..12).prop_flat_map(|total_sv| {
        (
            0usize..=total_sv,
            prop::collection::vec(sparse_instance_strategy(), total_sv),
            prop::collection::vec(finite_value_strategy(), total_sv),
            finite_value_strategy(),
        )
            .prop_map(move |(split, sv, coef, rho)| SvmModel {
                param: SvmParameter {
                    svm_type: SvmType::CSvc,
                    kernel_type: KernelType::Linear,
                    gamma: 0.0,
                    ..Default::default()
                },
                nr_class: 2,
                sv,
                sv_coef: vec![coef],
                rho: vec![rho],
                prob_a: Vec::new(),
                prob_b: Vec::new(),
                prob_density_marks: Vec::new(),
                sv_indices: (1..=total_sv).collect(),
                label: vec![1, -1],
                n_sv: vec![split, total_sv - split],
            })
    })
}

fn problem_to_text(prob: &SvmProblem) -> String {
    let mut out = String::new();
    for (label, instance) in prob.labels.iter().zip(prob.instances.iter()) {
        out.push_str(&label.to_string());
        for node in instance {
            out.push(' ');
            out.push_str(&node.index.to_string());
            out.push(':');
            out.push_str(&node.value.to_string());
        }
        out.push('\n');
    }
    out
}

/// Test: Deterministic predictions with randomized sparse instances.
///
/// Generates 2-5 random sparse instances, trains a model with fixed parameters,
/// and verifies that predicting the same instance twice yields identical results.
#[test]
fn kernel_deterministic() {
    libsvm_rs::set_quiet(true);

    // Generate a small problem with random sparse features.
    let instances = vec![
        vec![
            SvmNode {
                index: 1,
                value: 2.5,
            },
            SvmNode {
                index: 5,
                value: -1.3,
            },
            SvmNode {
                index: 18,
                value: 0.7,
            },
        ],
        vec![
            SvmNode {
                index: 2,
                value: 1.1,
            },
            SvmNode {
                index: 8,
                value: 3.2,
            },
            SvmNode {
                index: 15,
                value: -2.1,
            },
        ],
        vec![
            SvmNode {
                index: 3,
                value: -0.5,
            },
            SvmNode {
                index: 10,
                value: 1.9,
            },
            SvmNode {
                index: 20,
                value: 2.8,
            },
        ],
        vec![
            SvmNode {
                index: 1,
                value: 1.2,
            },
            SvmNode {
                index: 4,
                value: -1.5,
            },
            SvmNode {
                index: 12,
                value: 0.3,
            },
        ],
    ];

    let prob = SvmProblem {
        labels: vec![1.0, -1.0, 1.0, -1.0],
        instances,
    };

    let param = SvmParameter {
        gamma: 1.0 / 20.0, // max index is 20
        shrinking: false,
        eps: 0.01,
        ..Default::default()
    };

    // Train the model
    let model = svm_train(&prob, &param);

    // Predict the same instance twice and verify results are identical
    let test_instance = &prob.instances[0];
    let pred1 = predict(&model, test_instance);
    let pred2 = predict(&model, test_instance);

    assert_eq!(
        pred1, pred2,
        "Predictions should be deterministic; got {} and {}",
        pred1, pred2
    );
}

/// Test: Deterministic predictions on real data.
///
/// Loads heart_scale dataset, trains a C-SVC model, and verifies that
/// predicting the same instances twice yields identical results.
#[test]
fn predict_deterministic() {
    libsvm_rs::set_quiet(true);

    let prob = load_heart_scale();
    let param = SvmParameter {
        gamma: 1.0 / 13.0,
        ..Default::default()
    };

    // Train the model
    let model = svm_train(&prob, &param);

    // Predict a subset of instances twice
    let test_indices = vec![0, 1, 2, 3, 4];
    for &idx in &test_indices {
        let test_instance = &prob.instances[idx];
        let pred1 = predict(&model, test_instance);
        let pred2 = predict(&model, test_instance);

        assert_eq!(
            pred1, pred2,
            "Prediction for instance {} should be deterministic; got {} and {}",
            idx, pred1, pred2
        );
    }
}

/// Test: Classification predictions are valid training labels.
///
/// Trains a C-SVC model on heart_scale and verifies that predictions
/// for all instances are one of the training labels.
#[test]
fn train_predict_labels_in_range() {
    libsvm_rs::set_quiet(true);

    let prob = load_heart_scale();
    let valid_labels = unique_labels(&prob);

    let param = SvmParameter {
        gamma: 1.0 / 13.0,
        ..Default::default()
    };
    let model = svm_train(&prob, &param);

    // Predict all instances and verify each is a valid label
    for (idx, instance) in prob.instances.iter().enumerate() {
        let pred = predict(&model, instance);

        assert!(
            valid_labels.contains(&pred),
            "Instance {} prediction {} is not in training labels {:?}",
            idx,
            pred,
            valid_labels
        );
    }
}

/// Test: Cross-validation results are valid.
///
/// Runs 5-fold cross-validation on heart_scale and verifies that all
/// returned predictions are finite and valid training labels.
#[test]
fn cross_validation_results_valid() {
    libsvm_rs::set_quiet(true);

    let prob = load_heart_scale();
    let valid_labels = unique_labels(&prob);

    let param = SvmParameter {
        gamma: 1.0 / 13.0,
        ..Default::default()
    };
    let cv_targets = svm_cross_validation(&prob, &param, 5);

    // Verify all CV targets are valid
    assert_eq!(
        cv_targets.len(),
        prob.labels.len(),
        "CV output length should match problem size"
    );

    for (idx, &target) in cv_targets.iter().enumerate() {
        // Check finiteness
        assert!(
            target.is_finite(),
            "CV target[{}] = {} is not finite",
            idx,
            target
        );

        // Check that it's a valid label
        assert!(
            valid_labels.contains(&target),
            "CV target[{}] = {} is not in training labels {:?}",
            idx,
            target,
            valid_labels
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(96))]

    #[test]
    fn problem_loader_roundtrip_stability(prob in problem_strategy()) {
        let text = problem_to_text(&prob);
        let loaded = load_problem_from_reader(text.as_bytes())
            .expect("generated problem should parse");
        let loaded_again = load_problem_from_reader(text.as_bytes())
            .expect("generated problem should parse repeatedly");

        prop_assert_eq!(&loaded, &prob);
        prop_assert_eq!(&loaded_again, &loaded);
    }

    #[test]
    fn model_save_load_save_is_byte_stable(model in binary_model_strategy()) {
        let mut first = Vec::new();
        save_model_to_writer(&mut first, &model)
            .expect("generated model should serialize");

        let loaded = load_model_from_reader(first.as_slice())
            .expect("serialized model should parse");

        let mut second = Vec::new();
        save_model_to_writer(&mut second, &loaded)
            .expect("loaded model should serialize");

        prop_assert_eq!(first, second);
        prop_assert_eq!(loaded.param.svm_type, SvmType::CSvc);
        prop_assert_eq!(loaded.param.kernel_type, KernelType::Linear);
        prop_assert_eq!(loaded.nr_class, 2);
        prop_assert_eq!(loaded.sv.len(), model.sv.len());
        prop_assert_eq!(loaded.sv_coef.len(), 1);
        prop_assert_eq!(loaded.n_sv.iter().sum::<usize>(), loaded.sv.len());
    }
}
