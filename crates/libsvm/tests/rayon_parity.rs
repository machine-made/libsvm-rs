use libsvm_rs::cross_validation::svm_cross_validation;
use libsvm_rs::io::load_problem;
use libsvm_rs::{KernelType, SvmParameter, SvmType};
use std::path::PathBuf;

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("data")
}

// Digests depend on the platform-specific c_rand stream (glibc on Linux,
// BSD rand on macOS, fallback LCG elsewhere), so pinned values are asserted
// on Linux only; other platforms still exercise both code paths.
#[cfg(target_os = "linux")]
fn bits_digest(values: &[f64]) -> u64 {
    values.iter().fold(0xcbf29ce484222325, |hash, value| {
        let mut h = hash;
        for byte in value.to_bits().to_le_bytes() {
            h ^= u64::from(byte);
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    })
}

#[test]
fn cross_validation_to_bits_snapshots_match_with_and_without_rayon() {
    libsvm_rs::set_quiet(true);
    let heart = load_problem(&data_dir().join("heart_scale")).unwrap();
    let housing = load_problem(&data_dir().join("housing_scale")).unwrap();

    let classification = svm_cross_validation(
        &heart,
        &SvmParameter {
            svm_type: SvmType::CSvc,
            kernel_type: KernelType::Rbf,
            gamma: 1.0 / 13.0,
            c: 1.0,
            cache_size: 100.0,
            eps: 0.001,
            shrinking: true,
            ..Default::default()
        },
        5,
    );
    assert_eq!(classification.len(), heart.labels.len());
    #[cfg(target_os = "linux")]
    assert_eq!(bits_digest(&classification), 1_205_192_661_221_889_925);

    let svr = svm_cross_validation(
        &housing,
        &SvmParameter {
            svm_type: SvmType::EpsilonSvr,
            kernel_type: KernelType::Rbf,
            gamma: 1.0 / 13.0,
            c: 1.0,
            p: 0.1,
            cache_size: 100.0,
            eps: 0.001,
            shrinking: true,
            ..Default::default()
        },
        5,
    );
    assert_eq!(svr.len(), housing.labels.len());
    #[cfg(target_os = "linux")]
    assert_eq!(bits_digest(&svr), 13_577_796_025_173_739_283);

    let classification_probability = svm_cross_validation(
        &heart,
        &SvmParameter {
            svm_type: SvmType::CSvc,
            kernel_type: KernelType::Rbf,
            gamma: 1.0 / 13.0,
            c: 1.0,
            cache_size: 100.0,
            eps: 0.001,
            shrinking: true,
            probability: true,
            ..Default::default()
        },
        5,
    );
    assert_eq!(classification_probability.len(), heart.labels.len());
    #[cfg(target_os = "linux")]
    assert_eq!(
        bits_digest(&classification_probability),
        16_292_708_733_561_569_413
    );

    let svr_probability = svm_cross_validation(
        &housing,
        &SvmParameter {
            svm_type: SvmType::EpsilonSvr,
            kernel_type: KernelType::Rbf,
            gamma: 1.0 / 13.0,
            c: 1.0,
            p: 0.1,
            cache_size: 100.0,
            eps: 0.001,
            shrinking: true,
            probability: true,
            ..Default::default()
        },
        5,
    );
    assert_eq!(svr_probability.len(), housing.labels.len());
    #[cfg(target_os = "linux")]
    assert_eq!(bits_digest(&svr_probability), 2_107_691_266_322_774_531);
}
