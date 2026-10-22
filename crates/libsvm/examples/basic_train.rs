//! Basic end-to-end training example (train + evaluate + save).

use libsvm_rs::io::{load_problem, save_model};
use libsvm_rs::predict::predict;
use libsvm_rs::train::svm_train;
use libsvm_rs::{set_quiet, KernelType, SvmParameter, SvmType};
use std::path::Path;

fn main() {
    set_quiet(true);

    let problem = load_problem(Path::new("data/heart_scale")).expect("failed to load problem");

    let param = SvmParameter {
        svm_type: SvmType::CSvc,
        kernel_type: KernelType::Rbf,
        gamma: 1.0 / 13.0,
        ..Default::default()
    };

    let model = svm_train(&problem, &param);

    let correct = problem
        .instances
        .iter()
        .zip(problem.labels.iter())
        .filter(|(x, &y)| predict(&model, x) == y)
        .count();

    let acc = correct as f64 / problem.labels.len() as f64;
    save_model(Path::new("heart_scale.model"), &model).expect("failed to save model");

    println!("Training accuracy: {:.2}%", acc * 100.0);
}
