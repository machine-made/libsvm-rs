//! Simple grid search using 5-fold cross-validation.

use libsvm_rs::cross_validation::svm_cross_validation;
use libsvm_rs::io::load_problem;
use libsvm_rs::{set_quiet, KernelType, SvmParameter, SvmType};
use std::path::Path;

fn main() {
    set_quiet(true);

    let problem = load_problem(Path::new("data/heart_scale")).expect("failed to load problem");

    let cs = [0.1, 1.0, 10.0, 100.0];
    let gammas = [0.01, 0.1, 1.0];

    println!("{:>8} {:>8} {:>12}", "C", "gamma", "CV accuracy");

    let mut best_c = cs[0];
    let mut best_g = gammas[0];
    let mut best_acc = -1.0f64;

    for &c in &cs {
        for &gamma in &gammas {
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
                .filter(|(&p, &y)| p == y)
                .count();
            let acc = correct as f64 / problem.labels.len() as f64;

            println!("{:>8.1} {:>8.2} {:>11.2}%", c, gamma, acc * 100.0);

            if acc > best_acc {
                best_acc = acc;
                best_c = c;
                best_g = gamma;
            }
        }
    }

    println!(
        "\nBest: C={:.1}, gamma={:.2}, CV accuracy={:.2}%",
        best_c,
        best_g,
        best_acc * 100.0
    );
}
