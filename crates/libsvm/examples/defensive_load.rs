//! Load LIBSVM files with explicit resource caps.
//!
//! Run:
//!   cargo run -p libsvm-rs --example defensive_load

use libsvm_rs::io::{
    load_model_from_reader_with_options, load_problem_from_reader_with_options, LoadOptions,
};
use libsvm_rs::set_quiet;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    set_quiet(true);

    let options = LoadOptions {
        max_bytes: 10 * 1024 * 1024,
        max_line_len: 256 * 1024,
        max_sv: 500_000,
        max_nr_class: 1024,
        ..LoadOptions::default()
    };

    let problem_file = File::open(Path::new("data/heart_scale"))?;
    let problem = load_problem_from_reader_with_options(BufReader::new(problem_file), &options)?;

    let model_file = File::open(Path::new("data/heart_scale.model"))?;
    let model = load_model_from_reader_with_options(BufReader::new(model_file), &options)?;

    println!(
        "loaded {} instances and {} support vectors",
        problem.labels.len(),
        model.support_vector_count()
    );

    Ok(())
}
