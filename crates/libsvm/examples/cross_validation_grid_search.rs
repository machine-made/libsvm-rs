//! Simple grid search over C and gamma using 5-fold cross-validation.
//!
//! Run:
//!   cargo run -p libsvm-rs --example cross_validation_grid_search

use libsvm_rs::cross_validation::svm_cross_validation;
use libsvm_rs::io::load_problem;
use libsvm_rs::{set_quiet, KernelType, SvmParameter, SvmType};
use std::path::Path;

fn main() {
    set_quiet(true);

    let problem = load_problem(Path::new("data/heart_scale")).expect("failed to load dataset");

    let c_values = [0.1, 1.0, 10.0, 100.0];
    let gamma_values = [0.01, 0.05, 0.1, 0.5, 1.0];

    println!("{:>8} {:>8} {:>12}", "C", "gamma", "CV accuracy");

    let mut best = (c_values[0], gamma_values[0], -1.0f64);
    for &c in &c_values {
        for &gamma in &gamma_values {
            let param = SvmParameter {
                svm_type: SvmType::CSvc,
                kernel_type: KernelType::Rbf,
                c,
                gamma,
                ..Default::default()
            };

            let target = svm_cross_validation(&problem, &param, 5);
            let correct = target
                .iter()
                .zip(problem.labels.iter())
                .filter(|(pred, y)| **pred == **y)
                .count();
            let acc = correct as f64 / problem.labels.len() as f64;

            println!("{:>8.2} {:>8.2} {:>11.2}%", c, gamma, acc * 100.0);

            if acc > best.2 {
                best = (c, gamma, acc);
            }
        }
    }

    println!(
        "\nBest parameters: C={:.2}, gamma={:.2}, CV accuracy={:.2}%",
        best.0,
        best.1,
        best.2 * 100.0
    );
}
