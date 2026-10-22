#![cfg(feature = "serde")]

use std::path::PathBuf;

use libsvm_rs::io::{load_problem, save_model_to_writer};
use libsvm_rs::train::svm_train;
use libsvm_rs::{KernelType, SvmModel, SvmParameter, SvmType};

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("data")
}

fn model_text(model: &SvmModel) -> String {
    let mut bytes = Vec::new();
    save_model_to_writer(&mut bytes, model).unwrap();
    String::from_utf8(bytes).unwrap()
}

fn assert_f64_bits_stable(left: &SvmModel, right: &SvmModel) {
    assert_eq!(left.param.gamma.to_bits(), right.param.gamma.to_bits());
    assert_eq!(left.param.coef0.to_bits(), right.param.coef0.to_bits());
    assert_eq!(
        left.param.cache_size.to_bits(),
        right.param.cache_size.to_bits()
    );
    assert_eq!(left.param.eps.to_bits(), right.param.eps.to_bits());
    assert_eq!(left.param.c.to_bits(), right.param.c.to_bits());
    assert_eq!(left.param.nu.to_bits(), right.param.nu.to_bits());
    assert_eq!(left.param.p.to_bits(), right.param.p.to_bits());
    for (a, b) in left.sv.iter().flatten().zip(right.sv.iter().flatten()) {
        assert_eq!(a.value.to_bits(), b.value.to_bits());
    }
    for (a, b) in left
        .sv_coef
        .iter()
        .flatten()
        .zip(right.sv_coef.iter().flatten())
    {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    for (a, b) in left.rho.iter().zip(&right.rho) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    for (a, b) in left.prob_a.iter().zip(&right.prob_a) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    for (a, b) in left.prob_b.iter().zip(&right.prob_b) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

fn assert_json_roundtrip(model: SvmModel) {
    let json = serde_json::to_string(&model).unwrap();
    let decoded: SvmModel = serde_json::from_str(&json).unwrap();
    let json_again = serde_json::to_string(&decoded).unwrap();
    let decoded_again: SvmModel = serde_json::from_str(&json_again).unwrap();
    assert_f64_bits_stable(&decoded, &decoded_again);
    assert!(!model_text(&decoded).is_empty());
}

#[test]
fn enums_serialize_as_libsvm_integer_codes() {
    assert_eq!(serde_json::to_string(&SvmType::CSvc).unwrap(), "0");
    assert_eq!(serde_json::to_string(&SvmType::NuSvc).unwrap(), "1");
    assert_eq!(serde_json::to_string(&SvmType::OneClass).unwrap(), "2");
    assert_eq!(serde_json::to_string(&SvmType::EpsilonSvr).unwrap(), "3");
    assert_eq!(serde_json::to_string(&SvmType::NuSvr).unwrap(), "4");
    assert_eq!(serde_json::to_string(&KernelType::Linear).unwrap(), "0");
    assert_eq!(serde_json::to_string(&KernelType::Polynomial).unwrap(), "1");
    assert_eq!(serde_json::to_string(&KernelType::Rbf).unwrap(), "2");
    assert_eq!(serde_json::to_string(&KernelType::Sigmoid).unwrap(), "3");
    assert_eq!(
        serde_json::to_string(&KernelType::Precomputed).unwrap(),
        "4"
    );
}

#[test]
fn heart_scale_classification_roundtrips_through_json() {
    let problem = load_problem(&data_dir().join("heart_scale")).unwrap();
    let param = SvmParameter {
        gamma: 1.0 / 13.0,
        ..SvmParameter::default()
    };
    assert_json_roundtrip(svm_train(&problem, &param));
}

#[test]
fn heart_scale_probability_classification_roundtrips_through_json() {
    let problem = load_problem(&data_dir().join("heart_scale")).unwrap();
    let param = SvmParameter {
        gamma: 1.0 / 13.0,
        probability: true,
        ..SvmParameter::default()
    };
    assert_json_roundtrip(svm_train(&problem, &param));
}

#[test]
fn svr_roundtrips_through_json() {
    let problem = load_problem(&data_dir().join("housing_scale")).unwrap();
    let param = SvmParameter {
        svm_type: SvmType::EpsilonSvr,
        gamma: 1.0 / 13.0,
        p: 0.1,
        c: 1.0,
        ..SvmParameter::default()
    };
    assert_json_roundtrip(svm_train(&problem, &param));
}
