//! Train with probability estimates and print class probabilities.

use libsvm_rs::io::load_problem;
use libsvm_rs::predict::predict_probability;
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
        probability: true,
        ..Default::default()
    };

    let model = svm_train(&problem, &param);

    println!("Labels: {:?}", model.label);
    for i in 0..problem.instances.len().min(10) {
        let x = &problem.instances[i];
        let (pred, probs) = predict_probability(&model, x).expect("probability prediction failed");

        print!("i={:<3} true={:<3} pred={:<3}", i, problem.labels[i], pred);
        for (cls, p) in model.label.iter().zip(probs.iter()) {
            print!("  {}:{:.4}", cls, p);
        }
        println!();
    }
}
