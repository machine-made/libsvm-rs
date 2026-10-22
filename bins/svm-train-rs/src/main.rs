#![forbid(unsafe_code)]

use libsvm_rs::cross_validation::svm_cross_validation;
use libsvm_rs::io::{format_g, load_problem, save_model};
use libsvm_rs::train::svm_train;
use libsvm_rs::{
    accuracy_percentage, check_parameter, regression_metrics, KernelType, SvmParameter, SvmType,
};
use std::path::Path;
use std::process;

fn exit_with_help() -> ! {
    print!(
        "\
Usage: svm-train [options] training_set_file [model_file]
options:
-s svm_type : set type of SVM (default 0)
\t0 -- C-SVC\t\t(multi-class classification)
\t1 -- nu-SVC\t\t(multi-class classification)
\t2 -- one-class SVM
\t3 -- epsilon-SVR\t(regression)
\t4 -- nu-SVR\t\t(regression)
-t kernel_type : set type of kernel function (default 2)
\t0 -- linear: u'*v
\t1 -- polynomial: (gamma*u'*v + coef0)^degree
\t2 -- radial basis function: exp(-gamma*|u-v|^2)
\t3 -- sigmoid: tanh(gamma*u'*v + coef0)
\t4 -- precomputed kernel (kernel values in training_set_file)
-d degree : set degree in kernel function (default 3)
-g gamma : set gamma in kernel function (default 1/num_features)
-r coef0 : set coef0 in kernel function (default 0)
-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
-m cachesize : set cache memory size in MB (default 100)
-e epsilon : set tolerance of termination criterion (default 0.001)
-h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)
-b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
-wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)
-v n: n-fold cross validation mode
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

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut param = SvmParameter::default();
    let mut cross_validation = false;
    let mut nr_fold = 0usize;
    let mut quiet = false;

    let mut i = 1;
    while i < args.len() {
        if !args[i].starts_with('-') {
            break;
        }
        let flag = &args[i];

        // -q takes no argument
        if flag == "-q" {
            quiet = true;
            i += 1;
            continue;
        }

        // All other flags consume the next argument
        i += 1;
        let value = parse_flag_arg(&args, &mut i);

        if flag.starts_with("-w") && flag.len() > 2 {
            // -wi weight: class label is in flag[2..]
            let label: i32 = flag[2..].parse().unwrap_or_else(|_| {
                eprintln!("Unknown option: {}", flag);
                exit_with_help();
            });
            let weight: f64 = value.parse().unwrap_or_else(|_| {
                eprintln!("Unknown option: {}", flag);
                exit_with_help();
            });
            param.weight.push((label, weight));
        } else if flag.len() < 2 {
            eprintln!("Unknown option: {}", flag);
            exit_with_help();
        } else {
            match flag.as_bytes()[1] {
                b's' => {
                    param.svm_type = match value.parse::<i32>().unwrap_or(-1) {
                        0 => SvmType::CSvc,
                        1 => SvmType::NuSvc,
                        2 => SvmType::OneClass,
                        3 => SvmType::EpsilonSvr,
                        4 => SvmType::NuSvr,
                        _ => exit_with_help(),
                    };
                }
                b't' => {
                    param.kernel_type = match value.parse::<i32>().unwrap_or(-1) {
                        0 => KernelType::Linear,
                        1 => KernelType::Polynomial,
                        2 => KernelType::Rbf,
                        3 => KernelType::Sigmoid,
                        4 => KernelType::Precomputed,
                        _ => exit_with_help(),
                    };
                }
                b'd' => {
                    param.degree = value.parse().unwrap_or_else(|_| exit_with_help());
                }
                b'g' => {
                    param.gamma = value.parse().unwrap_or_else(|_| exit_with_help());
                }
                b'r' => {
                    param.coef0 = value.parse().unwrap_or_else(|_| exit_with_help());
                }
                b'c' => {
                    param.c = value.parse().unwrap_or_else(|_| exit_with_help());
                }
                b'n' => {
                    param.nu = value.parse().unwrap_or_else(|_| exit_with_help());
                }
                b'p' => {
                    param.p = value.parse().unwrap_or_else(|_| exit_with_help());
                }
                b'm' => {
                    param.cache_size = value.parse().unwrap_or_else(|_| exit_with_help());
                }
                b'e' => {
                    param.eps = value.parse().unwrap_or_else(|_| exit_with_help());
                }
                b'h' => {
                    param.shrinking = value.parse::<i32>().unwrap_or(1) != 0;
                }
                b'b' => {
                    param.probability = value.parse::<i32>().unwrap_or(0) != 0;
                }
                b'v' => {
                    cross_validation = true;
                    nr_fold = value.parse().unwrap_or(0);
                    if nr_fold < 2 {
                        eprintln!("n-fold cross validation: n must >= 2");
                        exit_with_help();
                    }
                }
                _ => {
                    eprintln!("Unknown option: {}", flag);
                    exit_with_help();
                }
            }
        }
    }

    // Remaining: training_set_file [model_file]
    if i >= args.len() {
        exit_with_help();
    }
    let input_file = &args[i];
    let model_file = if i + 1 < args.len() {
        args[i + 1].clone()
    } else {
        let base = Path::new(input_file)
            .file_name()
            .unwrap_or_default()
            .to_str()
            .unwrap_or("output");
        format!("{}.model", base)
    };

    // Load problem
    let problem = load_problem(Path::new(input_file)).unwrap_or_else(|e| {
        eprintln!("can't open input file {}: {}", input_file, e);
        process::exit(1);
    });

    // Auto-set gamma = 1/num_features
    if param.gamma == 0.0 {
        let max_index = problem
            .instances
            .iter()
            .flat_map(|inst| inst.iter())
            .map(|n| n.index)
            .max()
            .unwrap_or(0);
        if max_index > 0 {
            param.gamma = 1.0 / max_index as f64;
        }
    }

    // Quiet mode
    if quiet {
        libsvm_rs::set_quiet(true);
    }

    // Validate
    if let Err(e) = check_parameter(&problem, &param) {
        eprintln!("ERROR: {}", e);
        process::exit(1);
    }

    if cross_validation {
        do_cross_validation(&problem, &param, nr_fold);
    } else {
        let model = svm_train(&problem, &param);
        save_model(Path::new(&model_file), &model).unwrap_or_else(|e| {
            eprintln!("can't save model to file {}: {}", model_file, e);
            process::exit(1);
        });
    }
}

fn do_cross_validation(problem: &libsvm_rs::SvmProblem, param: &SvmParameter, nr_fold: usize) {
    let target = svm_cross_validation(problem, param, nr_fold);
    if matches!(param.svm_type, SvmType::EpsilonSvr | SvmType::NuSvr) {
        let (mse, r2) = regression_metrics(&target, &problem.labels);
        println!("Cross Validation Mean squared error = {}", format_g(mse));
        println!(
            "Cross Validation Squared correlation coefficient = {}",
            format_g(r2)
        );
    } else {
        let correct = accuracy_percentage(&target, &problem.labels);
        println!("Cross Validation Accuracy = {}%", format_g(correct));
    }
}
