use libsvm_rs::io::load_problem_from_reader;
use libsvm_rs::predict::predict;
use libsvm_rs::train::svm_train;
use libsvm_rs::{set_quiet, KernelType, SvmModel, SvmParameter, SvmProblem, SvmType};
use serde::Serialize;
use std::io::{BufReader, Cursor};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = performance)]
    fn now() -> f64;
}

#[derive(Serialize)]
struct WasmBenchResult {
    train_rows: usize,
    test_rows: usize,
    train_samples_ms: Vec<f64>,
    predict_samples_ms: Vec<f64>,
    accuracy: f64,
    predictions: Vec<f64>,
}

fn parse_problem(text: &str) -> SvmProblem {
    let cursor = Cursor::new(text.as_bytes());
    let reader = BufReader::new(cursor);
    load_problem_from_reader(reader).expect("failed to parse LIBSVM text")
}

fn run_predict(model: &SvmModel, test: &SvmProblem) -> Vec<f64> {
    test.instances.iter().map(|x| predict(model, x)).collect()
}

fn calc_accuracy(labels: &[f64], preds: &[f64]) -> f64 {
    if labels.is_empty() || labels.len() != preds.len() {
        return 0.0;
    }
    let ok = labels
        .iter()
        .zip(preds.iter())
        .filter(|(y, p)| (**y - **p).abs() < 1e-12)
        .count();
    ok as f64 / labels.len() as f64
}

fn now_ms() -> f64 {
    now()
}

#[wasm_bindgen]
pub fn benchmark_from_libsvm_text(
    train_text: &str,
    test_text: &str,
    warmup: u32,
    runs: u32,
) -> String {
    set_quiet(true);

    let train = parse_problem(train_text);
    let test = parse_problem(test_text);

    let param = SvmParameter {
        svm_type: SvmType::CSvc,
        kernel_type: KernelType::Rbf,
        c: 1.0,
        gamma: 1.0 / 13.0,
        ..Default::default()
    };

    for _ in 0..warmup {
        let model = svm_train(&train, &param);
        let _ = run_predict(&model, &test);
    }

    let run_count = runs.max(1);
    let mut train_samples_ms = Vec::with_capacity(run_count as usize);
    let mut predict_samples_ms = Vec::with_capacity(run_count as usize);
    let mut last_preds: Vec<f64> = Vec::new();

    for _ in 0..run_count {
        let t0 = now_ms();
        let model = svm_train(&train, &param);
        let t1 = now_ms();
        train_samples_ms.push(t1 - t0);

        let p0 = now_ms();
        let preds = run_predict(&model, &test);
        let p1 = now_ms();
        predict_samples_ms.push(p1 - p0);
        last_preds = preds;
    }

    let payload = WasmBenchResult {
        train_rows: train.labels.len(),
        test_rows: test.labels.len(),
        train_samples_ms,
        predict_samples_ms,
        accuracy: calc_accuracy(&test.labels, &last_preds),
        predictions: last_preds,
    };

    serde_json::to_string(&payload).expect("serialize wasm benchmark payload")
}
