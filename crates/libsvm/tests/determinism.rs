//! Compare model and prediction bits across threads and cross-validation pool sizes.

use libsvm_rs::predict::predict_values;
use libsvm_rs::train::svm_train;
use libsvm_rs::{KernelType, SvmNode, SvmParameter, SvmProblem, SvmType, check_parameter};
use std::error::Error;

type TestError = Box<dyn Error + Send + Sync>;

fn training_bits() -> Result<Vec<u64>, TestError> {
    let problem = SvmProblem {
        labels: vec![-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
        instances: [-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0]
            .into_iter()
            .map(|value| vec![SvmNode { index: 1, value }])
            .collect(),
    };
    let mut bits = Vec::new();
    for svm_type in [
        SvmType::CSvc,
        SvmType::NuSvc,
        SvmType::OneClass,
        SvmType::EpsilonSvr,
        SvmType::NuSvr,
    ] {
        let param = SvmParameter {
            svm_type,
            kernel_type: KernelType::Rbf,
            gamma: 0.5,
            cache_size: 1.0,
            ..Default::default()
        };
        check_parameter(&problem, &param)?;
        let model = svm_train(&problem, &param);
        // Include lengths so a changed shape cannot disappear through flattening.
        bits.push(u64::try_from(model.sv_coef.len())?);
        for row in &model.sv_coef {
            bits.push(u64::try_from(row.len())?);
            bits.extend(row.iter().map(|value| value.to_bits()));
        }
        bits.push(u64::try_from(model.rho.len())?);
        bits.extend(model.rho.iter().map(|value| value.to_bits()));
        for instance in &problem.instances {
            // Each problem has two classes or one regression/one-class decision.
            let mut decisions = [0.0];
            let prediction = predict_values(&model, instance, &mut decisions);
            bits.extend(decisions.into_iter().map(f64::to_bits));
            bits.push(prediction.to_bits());
            bits.push(libsvm_rs::predict::predict(&model, instance).to_bits());
        }
    }
    Ok(bits)
}

fn assert_bits_equal(expected: &[u64], actual: &[u64], context: &str) {
    assert_eq!(expected.len(), actual.len(), "{context}: value count");
    for (index, (&left, &right)) in expected.iter().zip(actual).enumerate() {
        assert_eq!(
            left, right,
            "{context}: first differing value {index}: {left:#018x} versus {right:#018x}"
        );
    }
}

#[cfg(feature = "rayon")]
fn cross_validation_bits(threads: usize) -> Result<Vec<u64>, TestError> {
    use libsvm_rs::cross_validation::svm_cross_validation;
    use libsvm_rs::io::load_problem;
    use std::path::Path;

    let data = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()?;
    let mut bits = Vec::new();
    for (file, svm_type) in [
        ("heart_scale", SvmType::CSvc),
        ("housing_scale", SvmType::EpsilonSvr),
    ] {
        let problem = load_problem(&data.join(file))?;
        let param = SvmParameter {
            svm_type,
            kernel_type: KernelType::Rbf,
            gamma: 1.0 / 13.0,
            ..Default::default()
        };
        check_parameter(&problem, &param)?;
        let values = pool.install(|| svm_cross_validation(&problem, &param, 5));
        assert_eq!(values.len(), problem.labels.len());
        bits.extend(values.into_iter().map(f64::to_bits));
    }
    Ok(bits)
}

#[cfg(feature = "rayon")]
fn fresh_process_bits(threads: usize) -> Result<Vec<u64>, TestError> {
    // The shuffle stream is process-global with no public reset API.
    // Each child starts at the same state and runs the same fixture order.
    let output = std::process::Command::new(std::env::current_exe()?)
        .args(["--exact", "bits_equal_for_1_and_4_threads", "--nocapture"])
        .env("LIBSVM_DETERMINISM_THREADS", threads.to_string())
        .output()?;
    assert!(
        output.status.success(),
        "{threads}-thread child failed: {}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout)?;
    let bits: Result<Vec<_>, _> = stdout
        .lines()
        .filter_map(|line| line.strip_prefix("determinism-bits:"))
        .map(|hex| u64::from_str_radix(hex, 16))
        .collect();
    let bits = bits?;
    assert!(
        !bits.is_empty(),
        "{threads}-thread child returned no values"
    );
    Ok(bits)
}

#[test]
fn bits_equal_for_1_and_4_threads() -> Result<(), TestError> {
    libsvm_rs::set_quiet(true);
    #[cfg(feature = "rayon")]
    if let Some(threads) = std::env::var_os("LIBSVM_DETERMINISM_THREADS") {
        let threads = threads
            .to_str()
            .ok_or("invalid thread count encoding")?
            .parse()?;
        for bits in cross_validation_bits(threads)? {
            println!("determinism-bits:{bits:016x}");
        }
        return Ok(());
    }

    std::thread::scope(|scope| -> Result<(), TestError> {
        let workers: Vec<_> = (0..4).map(|_| scope.spawn(training_bits)).collect();
        let expected = training_bits()?;
        for worker in workers {
            let actual = worker.join().map_err(|_| "training worker panicked")??;
            assert_bits_equal(&expected, &actual, "training and prediction");
        }
        Ok(())
    })?;

    #[cfg(feature = "rayon")]
    assert_bits_equal(
        &fresh_process_bits(1)?,
        &fresh_process_bits(4)?,
        "cross-validation: heart_scale then housing_scale",
    );
    Ok(())
}
