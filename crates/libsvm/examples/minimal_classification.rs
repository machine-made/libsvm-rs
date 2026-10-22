//! Minimal self-contained binary classification example.
//!
//! Run:
//!   cargo run -p libsvm-rs --example minimal_classification

use libsvm_rs::predict::predict;
use libsvm_rs::train::svm_train;
use libsvm_rs::{set_quiet, KernelType, SvmNode, SvmParameter, SvmProblem, SvmType};

fn point(x1: f64, x2: f64) -> Vec<SvmNode> {
    vec![
        SvmNode {
            index: 1,
            value: x1,
        },
        SvmNode {
            index: 2,
            value: x2,
        },
    ]
}

fn main() {
    set_quiet(true);

    // Tiny linearly-separable dataset in 2D.
    let problem = SvmProblem {
        labels: vec![1.0, 1.0, -1.0, -1.0],
        instances: vec![
            point(2.0, 2.0),
            point(2.5, 1.8),
            point(-2.0, -1.5),
            point(-2.4, -2.2),
        ],
    };

    let param = SvmParameter {
        svm_type: SvmType::CSvc,
        kernel_type: KernelType::Linear,
        c: 1.0,
        ..Default::default()
    };

    let model = svm_train(&problem, &param);

    let test = point(1.8, 1.9);
    let pred = predict(&model, &test);

    println!("Predicted label for [1.8, 1.9]: {pred}");

    let train_correct = problem
        .instances
        .iter()
        .zip(problem.labels.iter())
        .filter(|(x, y)| predict(&model, x) == **y)
        .count();
    println!(
        "Training accuracy: {}/{}",
        train_correct,
        problem.labels.len()
    );
}
