#![forbid(unsafe_code)]

use libsvm_rs::io::{format_17g, format_g, load_model, load_problem};
use libsvm_rs::predict::{predict, predict_probability};
use libsvm_rs::{regression_metrics, svm_check_probability_model, SvmType};
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::process;

fn exit_with_help() -> ! {
    print!(
        "\
Usage: svm-predict [options] test_file model_file output_file
options:
-b probability_estimates: whether to predict probability estimates, 0 or 1 (default 0); for one-class SVM only 0 is supported
-q : quiet mode (no outputs)
"
    );
    process::exit(1);
}

fn parse_flag_arg<'a>(args: &'a [String], i: &mut usize) -> &'a str {
    if *i >= args.len() {
        exit_with_help();
    }
    let value = &args[*i];
    *i += 1;
    value.as_str()
}

fn write_or_exit(result: io::Result<()>, output_file: &str) {
    result.unwrap_or_else(|e| {
        eprintln!("can't write output file {}: {}", output_file, e);
        process::exit(1);
    });
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut predict_prob = false;
    let mut quiet = false;

    let mut i = 1;
    while i < args.len() {
        if !args[i].starts_with('-') {
            break;
        }
        let flag = &args[i];

        if flag == "-q" {
            quiet = true;
            i += 1;
            continue;
        }

        i += 1;
        let value = parse_flag_arg(&args, &mut i);

        if flag.len() < 2 {
            eprintln!("Unknown option: {}", flag);
            exit_with_help();
        }
        match flag.as_bytes()[1] {
            b'b' => {
                predict_prob = value.parse::<i32>().unwrap_or(0) != 0;
            }
            _ => {
                eprintln!("Unknown option: {}", flag);
                exit_with_help();
            }
        }
    }

    // Need exactly 3 remaining args: test_file model_file output_file
    if i + 2 >= args.len() {
        exit_with_help();
    }
    let test_file = &args[i];
    let model_file = &args[i + 1];
    let output_file = &args[i + 2];

    // Quiet mode
    if quiet {
        libsvm_rs::set_quiet(true);
    }

    // Load model
    let model = load_model(Path::new(model_file)).unwrap_or_else(|e| {
        eprintln!("can't open model file {}: {}", model_file, e);
        process::exit(1);
    });

    // Check probability support
    if predict_prob {
        if !svm_check_probability_model(&model) {
            eprintln!("Model does not support probability estimates");
            process::exit(1);
        }
    } else if svm_check_probability_model(&model) && !quiet {
        eprintln!("Model supports probability estimates, but disabled in prediction.");
    }

    // Load test data
    let problem = load_problem(Path::new(test_file)).unwrap_or_else(|e| {
        eprintln!("can't open input file {}: {}", test_file, e);
        process::exit(1);
    });

    // Open output file
    let out = File::create(output_file).unwrap_or_else(|e| {
        eprintln!("can't open output file {}: {}", output_file, e);
        process::exit(1);
    });
    let mut out = BufWriter::new(out);

    let svm_type = model.param.svm_type;

    // Print probability header / SVR info
    if predict_prob {
        if matches!(svm_type, SvmType::EpsilonSvr | SvmType::NuSvr) {
            if !quiet {
                let sigma = model.svr_probability().unwrap_or(0.0);
                eprintln!(
                    "Prob. model for test data: target value = predicted value + z,\n\
                     z: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma={}",
                    format_g(sigma)
                );
            }
        } else if svm_type == SvmType::OneClass {
            write_or_exit(writeln!(out, "labels normal outlier"), output_file);
        } else {
            // Classification header
            write_or_exit(write!(out, "labels"), output_file);
            for &lab in &model.label {
                write_or_exit(write!(out, " {}", lab), output_file);
            }
            write_or_exit(writeln!(out), output_file);
        }
    }

    // Predict
    let mut predictions = Vec::with_capacity(problem.labels.len());

    let use_probability_output =
        predict_prob && matches!(svm_type, SvmType::CSvc | SvmType::NuSvc | SvmType::OneClass);

    for instance in &problem.instances {
        let predict_label = if use_probability_output {
            let (label, probs) = predict_probability(&model, instance).unwrap_or_else(|| {
                eprintln!("probability prediction failed");
                process::exit(1);
            });
            write_or_exit(write!(out, "{}", format_g(label)), output_file);
            for p in &probs {
                write_or_exit(write!(out, " {}", format_g(*p)), output_file);
            }
            write_or_exit(writeln!(out), output_file);
            label
        } else {
            let label = predict(&model, instance);
            write_or_exit(writeln!(out, "{}", format_17g(label)), output_file);
            label
        };

        predictions.push(predict_label);
    }

    if !quiet {
        if matches!(svm_type, SvmType::EpsilonSvr | SvmType::NuSvr) {
            let (mse, r2) = regression_metrics(&predictions, &problem.labels);
            eprintln!("Mean squared error = {} (regression)", format_g(mse));
            eprintln!(
                "Squared correlation coefficient = {} (regression)",
                format_g(r2)
            );
        } else {
            let total = problem.labels.len();
            let correct = predictions
                .iter()
                .zip(problem.labels.iter())
                .filter(|(&p, &l)| p == l)
                .count();
            eprintln!(
                "Accuracy = {}% ({}/{}) (classification)",
                format_g(100.0 * correct as f64 / total as f64),
                correct,
                total
            );
        }
    }
}
