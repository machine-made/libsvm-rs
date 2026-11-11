//! Load a saved model and compute accuracy on a dataset.
#![expect(clippy::expect_used, reason = "An example stops when a load fails")]
#![expect(
    clippy::cast_precision_loss,
    reason = "An example prints accuracy as a percent"
)]

use libsvm_rs::io::{load_model, load_problem};
use libsvm_rs::predict::predict;
use libsvm_rs::set_quiet;
use std::path::Path;

#[expect(
    clippy::float_cmp,
    reason = "Predicted labels are compared with class labels copied from the input"
)]
fn main() {
    set_quiet(true);

    let args: Vec<String> = std::env::args().collect();
    let model_path = args
        .get(1)
        .map_or("heart_scale.model", std::string::String::as_str);
    let test_path = args
        .get(2)
        .map_or("data/heart_scale", std::string::String::as_str);

    let model = load_model(Path::new(model_path)).expect("failed to load model");
    let problem = load_problem(Path::new(test_path)).expect("failed to load test data");

    let correct = problem
        .instances
        .iter()
        .zip(problem.labels.iter())
        .filter(|&(x, &y)| predict(&model, x) == y)
        .count();

    let acc = correct as f64 / problem.labels.len() as f64;
    println!(
        "Accuracy: {:.2}% ({}/{})",
        acc * 100.0,
        correct,
        problem.labels.len()
    );
}
