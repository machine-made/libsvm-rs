use criterion::{criterion_group, criterion_main, Criterion};
use libsvm_rs::io::load_problem;
use libsvm_rs::predict::predict;
use libsvm_rs::train::svm_train;
use libsvm_rs::{KernelType, SvmParameter, SvmType};
use std::hint::black_box;
use std::path::Path;

fn load_heart_scale() -> libsvm_rs::SvmProblem {
    let path = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../data/heart_scale"
    ));
    load_problem(path).expect("failed to load heart_scale dataset")
}

fn bench_train_rbf(c: &mut Criterion) {
    libsvm_rs::set_quiet(true);
    let problem = load_heart_scale();
    let param = SvmParameter {
        svm_type: SvmType::CSvc,
        kernel_type: KernelType::Rbf,
        gamma: 1.0 / 13.0,
        ..Default::default()
    };

    c.bench_function("train_rbf", |b| {
        b.iter(|| svm_train(black_box(&problem), black_box(&param)))
    });
}

fn bench_train_linear(c: &mut Criterion) {
    libsvm_rs::set_quiet(true);
    let problem = load_heart_scale();
    let param = SvmParameter {
        svm_type: SvmType::CSvc,
        kernel_type: KernelType::Linear,
        ..Default::default()
    };

    c.bench_function("train_linear", |b| {
        b.iter(|| svm_train(black_box(&problem), black_box(&param)))
    });
}

fn bench_predict(c: &mut Criterion) {
    libsvm_rs::set_quiet(true);
    let problem = load_heart_scale();
    let param = SvmParameter {
        gamma: 1.0 / 13.0,
        ..Default::default()
    };
    let model = svm_train(&problem, &param);

    c.bench_function("predict_all", |b| {
        b.iter(|| {
            for instance in &problem.instances {
                let _ = predict(black_box(&model), black_box(instance));
            }
        })
    });
}

fn bench_train_with_probability(c: &mut Criterion) {
    libsvm_rs::set_quiet(true);
    let problem = load_heart_scale();
    let param = SvmParameter {
        gamma: 1.0 / 13.0,
        probability: true,
        ..Default::default()
    };

    c.bench_function("train_with_probability", |b| {
        b.iter(|| svm_train(black_box(&problem), black_box(&param)))
    });
}

fn load_housing_scale() -> libsvm_rs::SvmProblem {
    let path = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../data/housing_scale"
    ));
    load_problem(path).expect("failed to load housing_scale dataset")
}

fn bench_svr(c: &mut Criterion, name: &str, svm_type: SvmType, kernel_type: KernelType) {
    let problem = load_housing_scale();
    let param = SvmParameter {
        svm_type,
        kernel_type,
        ..Default::default()
    };

    c.bench_function(name, |b| {
        b.iter(|| svm_train(black_box(&problem), black_box(&param)))
    });
}

fn bench_svr_epsilon_linear(c: &mut Criterion) {
    libsvm_rs::set_quiet(true);
    bench_svr(
        c,
        "svr_epsilon_linear",
        SvmType::EpsilonSvr,
        KernelType::Linear,
    );
}

fn bench_svr_epsilon_rbf(c: &mut Criterion) {
    libsvm_rs::set_quiet(true);
    bench_svr(c, "svr_epsilon_rbf", SvmType::EpsilonSvr, KernelType::Rbf);
}

fn bench_svr_nu_linear(c: &mut Criterion) {
    libsvm_rs::set_quiet(true);
    bench_svr(c, "svr_nu_linear", SvmType::NuSvr, KernelType::Linear);
}

fn bench_svr_nu_rbf(c: &mut Criterion) {
    libsvm_rs::set_quiet(true);
    bench_svr(c, "svr_nu_rbf", SvmType::NuSvr, KernelType::Rbf);
}

criterion_group!(
    benches,
    bench_train_rbf,
    bench_train_linear,
    bench_predict,
    bench_train_with_probability,
    bench_svr_epsilon_linear,
    bench_svr_epsilon_rbf,
    bench_svr_nu_linear,
    bench_svr_nu_rbf
);
criterion_main!(benches);
