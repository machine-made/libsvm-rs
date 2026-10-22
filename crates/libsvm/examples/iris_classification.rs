//! Classic Iris workflow example using LIBSVM-format iris dataset.
//!
//! Run:
//!   cargo run -p libsvm-rs --example iris_classification

use libsvm_rs::io::load_problem;
use libsvm_rs::predict::predict;
use libsvm_rs::train::svm_train;
use libsvm_rs::{set_quiet, KernelType, SvmParameter, SvmType};
use std::path::Path;

fn main() {
    set_quiet(true);

    let problem = load_problem(Path::new("data/iris.scale")).expect("failed to load iris dataset");

    let param = SvmParameter {
        svm_type: SvmType::CSvc,
        kernel_type: KernelType::Rbf,
        gamma: 1.0 / 4.0, // iris has 4 features
        c: 1.0,
        ..Default::default()
    };

    let model = svm_train(&problem, &param);

    let correct = problem
        .instances
        .iter()
        .zip(problem.labels.iter())
        .filter(|(x, y)| predict(&model, x) == **y)
        .count();

    let acc = correct as f64 / problem.labels.len() as f64;

    println!(
        "Iris training accuracy: {:.2}% ({}/{})",
        acc * 100.0,
        correct,
        problem.labels.len()
    );
    println!("Classes seen: {:?}", model.label);
}
