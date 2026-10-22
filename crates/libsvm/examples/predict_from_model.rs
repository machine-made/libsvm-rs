//! Load a saved model and compute accuracy on a dataset.

use libsvm_rs::io::{load_model, load_problem};
use libsvm_rs::predict::predict;
use libsvm_rs::set_quiet;
use std::path::Path;

fn main() {
    set_quiet(true);

    let args: Vec<String> = std::env::args().collect();
    let model_path = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("heart_scale.model");
    let test_path = args
        .get(2)
        .map(|s| s.as_str())
        .unwrap_or("data/heart_scale");

    let model = load_model(Path::new(model_path)).expect("failed to load model");
    let problem = load_problem(Path::new(test_path)).expect("failed to load test data");

    let correct = problem
        .instances
        .iter()
        .zip(problem.labels.iter())
        .filter(|(x, &y)| predict(&model, x) == y)
        .count();

    let acc = correct as f64 / problem.labels.len() as f64;
    println!(
        "Accuracy: {:.2}% ({}/{})",
        acc * 100.0,
        correct,
        problem.labels.len()
    );
}
