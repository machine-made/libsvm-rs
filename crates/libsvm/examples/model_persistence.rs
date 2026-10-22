//! Train, save, load, and predict with a persisted model.
//!
//! Run:
//!   cargo run -p libsvm-rs --example model_persistence

use libsvm_rs::io::{load_model, load_problem, save_model};
use libsvm_rs::predict::predict;
use libsvm_rs::train::svm_train;
use libsvm_rs::{set_quiet, KernelType, SvmParameter, SvmType};
use std::path::Path;

fn main() {
    set_quiet(true);

    let data_path = Path::new("data/heart_scale");
    let model_path = Path::new("data/heart_scale.persisted.model");

    let problem = load_problem(data_path).expect("failed to load dataset");

    let param = SvmParameter {
        svm_type: SvmType::CSvc,
        kernel_type: KernelType::Rbf,
        gamma: 1.0 / 13.0,
        c: 1.0,
        ..Default::default()
    };

    let model = svm_train(&problem, &param);
    save_model(model_path, &model).expect("failed to save model");

    let loaded = load_model(model_path).expect("failed to load model");

    let first = &problem.instances[0];
    let p1 = predict(&model, first);
    let p2 = predict(&loaded, first);

    println!("Original model prediction: {p1}");
    println!("Loaded model prediction:   {p2}");
    println!("Model persisted at: {}", model_path.display());
}
