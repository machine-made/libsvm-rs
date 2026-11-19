//! Prediction functions matching the original LIBSVM.
//!
//! Provides `predict`, `predict_values`, and `predict_probability` for
//! all SVM types:
//! - Classification (C-SVC, ν-SVC): one-vs-one voting + optional probability
//! - One-class SVM: sign of decision value + optional density probability
//! - Regression (ε-SVR, ν-SVR): continuous output

#![cfg_attr(
    test,
    expect(
        clippy::unwrap_used,
        reason = "Tests unwrap fixture loading and checked assertions"
    )
)]

use crate::kernel::k_function;
use crate::probability::{multiclass_probability, predict_one_class_probability, sigmoid_predict};
use crate::types::{SvmModel, SvmNode, SvmType};

/// Compute decision values and return the predicted label/value.
///
/// For classification, `dec_values` receives `nr_class * (nr_class - 1) / 2`
/// pairwise decision values. For regression/one-class, a single value.
///
/// Returns the predicted label (classification) or function value (regression).
///
/// Matches LIBSVM's `svm_predict_values`.
/// # Panics
///
/// Panics if model arrays are inconsistent, class counts overflow, or the decision buffer is too short.
pub fn predict_values(model: &SvmModel, x: &[SvmNode], dec_values: &mut [f64]) -> f64 {
    let value = try_predict_values(model, x, dec_values);
    #[expect(
        clippy::expect_used,
        reason = "try_predict_values checks model slices, pair counts and the decision buffer"
    )]
    value.expect("prediction requires consistent model arrays and a sufficient decision buffer; give dec_values one slot per class pair and match n_sv to the support vectors")
}

fn try_predict_values(model: &SvmModel, x: &[SvmNode], dec_values: &mut [f64]) -> Option<f64> {
    match model.param.svm_type {
        SvmType::OneClass | SvmType::EpsilonSvr | SvmType::NuSvr => {
            let sv_coef = model.sv_coef.first()?.get(..model.sv.len())?;
            let mut sum = 0.0;
            for (&coef, sv) in sv_coef.iter().zip(&model.sv) {
                sum += coef * k_function(x, sv, &model.param);
            }
            sum -= model.rho.first()?;
            *dec_values.first_mut()? = sum;

            if model.param.svm_type == SvmType::OneClass {
                Some(if sum > 0.0 { 1.0 } else { -1.0 })
            } else {
                Some(sum)
            }
        }
        SvmType::CSvc | SvmType::NuSvc => {
            let nr_class = model.nr_class;
            let l = model.sv.len();

            // Compute kernel values for all SVs
            let kvalue: Vec<f64> = model
                .sv
                .iter()
                .map(|sv| k_function(x, sv, &model.param))
                .collect();

            let counts = model.n_sv.get(..nr_class)?;
            let labels = model.label.get(..nr_class)?;
            let mut start = vec![0usize; nr_class];
            let mut cumulative = 0usize;
            for (slot, &n) in start.iter_mut().zip(counts) {
                *slot = cumulative;
                cumulative = cumulative.checked_add(n)?;
            }
            if cumulative != l || nr_class == 0 {
                return None;
            }
            let pairs = pair_count(nr_class)?;
            let mut rhos = model.rho.get(..pairs)?.iter();
            let mut outputs = dec_values.get_mut(..pairs)?.iter_mut();
            let mut vote = vec![0usize; nr_class];
            for i in 0..nr_class {
                for j in i.checked_add(1)?..nr_class {
                    let mut sum = 0.0;
                    let si = *start.get(i)?;
                    let sj = *start.get(j)?;
                    let ci = *counts.get(i)?;
                    let cj = *counts.get(j)?;
                    let end_i = si.checked_add(ci)?;
                    let end_j = sj.checked_add(cj)?;
                    let coef1 = model.sv_coef.get(j.checked_sub(1)?)?.get(si..end_i)?;
                    let coef2 = model.sv_coef.get(i)?.get(sj..end_j)?;
                    let kernels_i = kvalue.get(si..end_i)?;
                    let kernels_j = kvalue.get(sj..end_j)?;
                    for (&coef, &kernel) in coef1.iter().zip(kernels_i) {
                        sum += coef * kernel;
                    }
                    for (&coef, &kernel) in coef2.iter().zip(kernels_j) {
                        sum += coef * kernel;
                    }
                    sum -= rhos.next()?;
                    *outputs.next()? = sum;
                    let winner = if sum > 0.0 { i } else { j };
                    let votes = vote.get_mut(winner)?;
                    *votes = votes.checked_add(1)?;
                }
            }

            // Find class with most votes
            let mut vote_max_idx = 0;
            let mut most_votes = *vote.first()?;
            for (i, &count) in vote.iter().enumerate().skip(1) {
                if count > most_votes {
                    vote_max_idx = i;
                    most_votes = count;
                }
            }

            Some(f64::from(*labels.get(vote_max_idx)?))
        }
    }
}

/// Predict the label/value for a single instance.
///
/// Convenience wrapper around `predict_values` that allocates the
/// decision values buffer internally. Matches LIBSVM's `svm_predict`.
///
/// ```
/// use libsvm_rs::predict::predict;
/// use libsvm_rs::train::svm_train;
/// use libsvm_rs::{KernelType, SvmNode, SvmParameterBuilder, SvmProblem, SvmType};
///
/// let problem = SvmProblem {
///     labels: vec![-1.0, -1.0, 1.0, 1.0],
///     instances: vec![
///         vec![SvmNode { index: 1, value: -2.0 }],
///         vec![SvmNode { index: 1, value: -1.0 }],
///         vec![SvmNode { index: 1, value: 1.0 }],
///         vec![SvmNode { index: 1, value: 2.0 }],
///     ],
/// };
/// let param = SvmParameterBuilder::new()
///     .svm_type(SvmType::CSvc)
///     .kernel_type(KernelType::Linear)
///     .build()?;
/// let model = svm_train(&problem, &param);
///
/// assert_eq!(predict(&model, &[SvmNode { index: 1, value: 1.5 }]), 1.0);
/// # Ok::<(), libsvm_rs::SvmError>(())
/// ```
/// # Panics
///
/// Panics if model arrays are inconsistent or class pair counts overflow.
#[must_use]
pub fn predict(model: &SvmModel, x: &[SvmNode]) -> f64 {
    let value = try_predict(model, x);
    #[expect(
        clippy::expect_used,
        reason = "try_predict checks class pair arithmetic and delegates model slice checks"
    )]
    value.expect("prediction requires consistent model arrays and representable class pair counts; load the model with load_model or make nr_class, rho, n_sv and sv_coef agree")
}

fn pair_count(classes: usize) -> Option<usize> {
    Some(classes.checked_mul(classes.checked_sub(1)?)? / 2)
}

fn try_predict(model: &SvmModel, x: &[SvmNode]) -> Option<f64> {
    let n = match model.param.svm_type {
        SvmType::OneClass | SvmType::EpsilonSvr | SvmType::NuSvr => 1,
        SvmType::CSvc | SvmType::NuSvc => pair_count(model.nr_class)?,
    };
    let mut dec_values = vec![0.0; n];
    try_predict_values(model, x, &mut dec_values)
}

/// Predict with probability estimates.
///
/// Returns `Some((label, probs))` where `probs[i]` is the estimated
/// probability of class `model.label[i]`. Returns `None` when the
/// model was not trained with probability support.
///
/// - **C-SVC / ν-SVC**: requires `model.prob_a` and `model.prob_b`.
///   Uses Platt scaling on pairwise decision values, then
///   `multiclass_probability` for k > 2.
/// - **One-class**: requires `model.prob_density_marks`.
///   Returns `[p, 1-p]` via density-mark lookup.
/// - **SVR**: probability prediction is not supported (returns `None`).
///
/// Matches LIBSVM's `svm_predict_probability`.
/// # Panics
///
/// Panics if a probability model has inconsistent arrays or class pair counts overflow.
#[must_use]
pub fn predict_probability(model: &SvmModel, x: &[SvmNode]) -> Option<(f64, Vec<f64>)> {
    let value = try_predict_probability(model, x);
    #[expect(clippy::expect_used, reason = "try_predict_probability checks probability arrays, pair counts and model slices")]
    value.expect("probability prediction requires consistent model arrays and representable class pair counts; make prob_a, prob_b, rho and n_sv agree with nr_class")
}

fn try_predict_probability(
    model: &SvmModel,
    x: &[SvmNode],
) -> Result<Option<(f64, Vec<f64>)>, &'static str> {
    (|| {
        match model.param.svm_type {
            SvmType::CSvc | SvmType::NuSvc
                if !model.prob_a.is_empty() && !model.prob_b.is_empty() =>
            {
                let nr_class = model.nr_class;
                let n_pairs = pair_count(nr_class)?;
                let mut dec_values = vec![0.0; n_pairs];
                try_predict_values(model, x, &mut dec_values)?;

                let min_prob = 1e-7;

                // Build pairwise probability matrix
                let mut pairwise = vec![vec![0.0; nr_class]; nr_class];
                let parameters = model
                    .prob_a
                    .get(..n_pairs)?
                    .iter()
                    .zip(model.prob_b.get(..n_pairs)?);
                let mut values = dec_values.iter().zip(parameters);
                for i in 0..nr_class {
                    let next = i.checked_add(1)?;
                    let (before, after) = pairwise.split_at_mut_checked(next)?;
                    let row_i = before.last_mut()?;
                    for (slot, row_j) in row_i.get_mut(next..)?.iter_mut().zip(after) {
                        let (&decision, (&a, &b)) = values.next()?;
                        let p = sigmoid_predict(decision, a, b)
                            .max(min_prob)
                            .min(1.0 - min_prob);
                        *slot = p;
                        // pairwise has nr_class columns per row, and i is below nr_class.
                        #[expect(clippy::indexing_slicing, reason = "pairwise is allocated as nr_class by nr_class; i is below nr_class")]
                        let opposite = &mut row_j[i];
                        *opposite = 1.0 - p;
                    }
                }

                let mut prob_estimates = vec![0.0; nr_class];
                if nr_class == 2 {
                    let [first, second] = prob_estimates.as_mut_slice() else {
                        return None;
                    };
                    *first = *pairwise.first()?.get(1)?;
                    *second = *pairwise.get(1)?.first()?;
                } else {
                    multiclass_probability(nr_class, &pairwise, &mut prob_estimates);
                }

                // Find class with highest probability
                let mut best = 0;
                let mut best_value = *prob_estimates.first()?;
                for (i, &value) in prob_estimates.iter().enumerate().skip(1) {
                    if value > best_value {
                        best = i;
                        best_value = value;
                    }
                }
                Some(Some((f64::from(*model.label.get(best)?), prob_estimates)))
            }

            SvmType::OneClass if !model.prob_density_marks.is_empty() => {
                let mut dec_value = [0.0];
                let pred_result = try_predict_values(model, x, &mut dec_value)?;
                let p = predict_one_class_probability(&model.prob_density_marks, dec_value[0]);
                Some(Some((pred_result, vec![p, 1.0 - p])))
            }

            _ => Some(None),
        }
    })()
    .ok_or("inconsistent probability model arrays or class pair counts; give prob_a and prob_b one value per class pair")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::load_model;
    use crate::io::load_problem;
    use std::path::PathBuf;

    #[test]
    fn predict_vote_tie_takes_the_first_label() {
        let model = SvmModel {
            param: crate::SvmParameter::default(),
            nr_class: 3,
            sv: Vec::new(),
            sv_coef: vec![Vec::new(), Vec::new()],
            rho: vec![-1.0, 1.0, -1.0],
            prob_a: Vec::new(),
            prob_b: Vec::new(),
            prob_density_marks: Vec::new(),
            sv_indices: Vec::new(),
            label: vec![7, -3, 2],
            n_sv: vec![0, 0, 0],
        };
        let mut decisions = [0.0; 3];
        let label = predict_values(&model, &[], &mut decisions);
        // The pairs (0, 1), (0, 2), (1, 2) vote for classes 0, 2, 1.
        // The comparisons are exact (zero tolerance). The model has no support vectors,
        // so each decision value is 0.0 - rho with rho = +-1.0, and a label is an i32
        // converted to f64. Neither step rounds.
        assert_eq!(
            decisions.map(f64::to_bits),
            [1.0_f64, -1.0, 1.0].map(f64::to_bits)
        );
        assert_eq!(
            label.to_bits(),
            7.0_f64.to_bits(),
            "one vote per class must keep the first label"
        );
        assert_eq!(predict(&model, &[]).to_bits(), 7.0_f64.to_bits());
    }

    #[test]
    fn pair_count_rejects_overflow() {
        assert!(pair_count(0).is_none());
        assert!(pair_count(usize::MAX).is_none());
        assert_eq!(pair_count(3), Some(3));
    }

    #[test]
    fn try_predict_values_rejects_short_buffer_and_coefficients() {
        let mut model = load_model(&data_dir().join("heart_scale.model")).unwrap();
        assert!(try_predict_values(&model, &[], &mut []).is_none());
        model.sv_coef.clear();
        assert!(try_predict_values(&model, &[], &mut [0.0]).is_none());
    }

    #[test]
    fn try_predict_rejects_inconsistent_model() {
        let mut model = load_model(&data_dir().join("heart_scale.model")).unwrap();
        model.rho.clear();
        assert!(try_predict(&model, &[]).is_none());
    }

    #[test]
    fn try_predict_probability_rejects_short_probability_arrays() {
        let mut model = load_model(&data_dir().join("heart_scale.model")).unwrap();
        model.nr_class = 3;
        model.label.push(3);
        model.n_sv.push(0);
        model.rho.resize(3, 0.0);
        model.sv_coef.push(model.sv_coef.first().unwrap().clone());
        model.prob_a = vec![0.0];
        model.prob_b = vec![0.0];
        assert!(try_predict_probability(&model, &[]).is_err());
    }

    fn data_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("data")
    }

    #[test]
    #[expect(
        clippy::cast_precision_loss,
        reason = "C converts the same integer counts to double at these points"
    )]
    #[expect(
        clippy::float_cmp,
        reason = "Predicted labels are compared with class labels copied from the input"
    )]
    fn predict_heart_scale() {
        // Load model trained by C LIBSVM and predict on training data
        let model = load_model(&data_dir().join("heart_scale.model")).unwrap();
        let problem = load_problem(&data_dir().join("heart_scale")).unwrap();

        let mut correct = 0;
        for (i, instance) in problem.instances.iter().enumerate() {
            let pred = predict(&model, instance);
            if pred == *problem.labels.get(i).unwrap() {
                correct = i32::checked_add(correct, 1).unwrap();
            }
        }

        let accuracy = f64::from(correct) / problem.labels.len() as f64;
        // C LIBSVM gets ~86.67% accuracy on training set with default params
        assert!(
            accuracy > 0.85,
            "accuracy {:.2}% too low (expected >85%)",
            accuracy * 100.0
        );
    }

    #[test]
    fn predict_values_binary() {
        let model = load_model(&data_dir().join("heart_scale.model")).unwrap();
        let problem = load_problem(&data_dir().join("heart_scale")).unwrap();

        // For binary classification, there's exactly 1 decision value
        let mut dec_values = vec![0.0; 1];
        let label = predict_values(&model, problem.instances.first().unwrap(), &mut dec_values);

        // Decision value should be non-zero
        // Threshold 1e-10: only separates a real decision value from rounding noise near 1e-15.
        assert!(dec_values.first().unwrap().abs() > 1e-10);
        // Label should match what predict returns
        assert_eq!(label, predict(&model, problem.instances.first().unwrap()));
    }

    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "Predicted labels are the exact class values 1 and -1"
    )]
    fn predict_probability_binary() {
        let problem = load_problem(&data_dir().join("heart_scale")).unwrap();
        let param = crate::types::SvmParameter {
            svm_type: SvmType::CSvc,
            kernel_type: crate::types::KernelType::Rbf,
            gamma: 1.0 / 13.0,
            c: 1.0,
            cache_size: 100.0,
            eps: 0.001,
            shrinking: true,
            probability: true,
            ..Default::default()
        };

        let model = crate::train::svm_train(&problem, &param);
        assert_ne!(model.prob_a.len(), 0);

        for instance in &problem.instances {
            let result = predict_probability(&model, instance);
            assert!(result.is_some(), "should return probability");
            let (label, probs) = result.unwrap();
            assert!(label == 1.0 || label == -1.0);
            assert_eq!(probs.len(), 2);
            let sum: f64 = probs.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "probs sum to {sum}, expected 1.0");
            for &p in &probs {
                assert!((0.0..=1.0).contains(&p), "prob {p} out of [0,1]");
            }
        }
    }

    #[test]
    fn predict_probability_multiclass() {
        let problem = load_problem(&data_dir().join("iris.scale")).unwrap();
        let param = crate::types::SvmParameter {
            svm_type: SvmType::CSvc,
            kernel_type: crate::types::KernelType::Rbf,
            gamma: 0.25,
            c: 1.0,
            cache_size: 100.0,
            eps: 0.001,
            shrinking: true,
            probability: true,
            ..Default::default()
        };

        let model = crate::train::svm_train(&problem, &param);
        assert_eq!(model.nr_class, 3);
        assert_eq!(model.prob_a.len(), 3); // 3 pairs

        for instance in problem.instances.iter().take(10) {
            let result = predict_probability(&model, instance);
            assert!(result.is_some());
            let (_label, probs) = result.unwrap();
            assert_eq!(probs.len(), 3);
            let sum: f64 = probs.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "probs sum to {sum}, expected 1.0");
        }
    }

    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "Predicted labels are compared with the labels the C tool wrote"
    )]
    fn predict_matches_c_svm_predict() {
        // Run C svm-predict and compare outputs
        // First, let's verify our predictions match by checking a few specific instances
        let model = load_model(&data_dir().join("heart_scale.model")).unwrap();
        let problem = load_problem(&data_dir().join("heart_scale")).unwrap();

        // Run C svm-predict to get reference predictions
        let c_predict = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("vendor")
            .join("libsvm")
            .join("svm-predict");

        if !c_predict.exists() {
            // Skip if C binary not compiled
            return;
        }

        // Write predictions to a temp file
        let output_path = data_dir().join("heart_scale.predict_test");
        let status = std::process::Command::new(&c_predict)
            .args([
                data_dir().join("heart_scale").to_str().unwrap(),
                data_dir().join("heart_scale.model").to_str().unwrap(),
                output_path.to_str().unwrap(),
            ])
            .output();

        if let Ok(output) = status {
            if output.status.success() {
                let c_preds: Vec<f64> = std::fs::read_to_string(&output_path)
                    .unwrap()
                    .lines()
                    .filter(|l| !l.is_empty())
                    .map(|l| l.trim().parse().unwrap())
                    .collect();

                assert_eq!(c_preds.len(), problem.labels.len());

                let mut mismatches = 0;
                for (i, instance) in problem.instances.iter().enumerate() {
                    let rust_pred = predict(&model, instance);
                    if rust_pred != *c_preds.get(i).unwrap() {
                        mismatches = i32::checked_add(mismatches, 1).unwrap();
                    }
                }

                assert_eq!(
                    mismatches, 0,
                    "{mismatches} predictions differ between Rust and C"
                );

                // Clean up
                let _ = std::fs::remove_file(&output_path);
            }
        }
    }
}
