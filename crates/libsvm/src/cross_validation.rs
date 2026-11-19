//! Cross-validation for SVM models.
//!
//! Provides stratified cross-validation for classification and simple
//! random-split cross-validation for regression/one-class problems.
//! Matches LIBSVM's `svm_cross_validation` (svm.cpp:2437–2556).

#![cfg_attr(
    test,
    expect(
        clippy::unwrap_used,
        reason = "Tests unwrap fixture loading and checked assertions"
    )
)]

use crate::predict::{predict, predict_probability};
use crate::train::svm_train;
use crate::types::{SvmModel, SvmNode, SvmParameter, SvmProblem, SvmType};
use crate::util::{group_classes, shuffle_range};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

fn predict_cv_target(model: &SvmModel, param: &SvmParameter, x: &[SvmNode]) -> f64 {
    if param.probability && matches!(param.svm_type, SvmType::CSvc | SvmType::NuSvc) {
        predict_probability(model, x).map_or_else(|| predict(model, x), |(label, _)| label)
    } else {
        predict(model, x)
    }
}

fn fold_starts(fold_count: &[usize]) -> Option<Vec<usize>> {
    let capacity = fold_count.len().checked_add(1)?;
    let mut starts = Vec::with_capacity(capacity);
    let mut total = 0usize;
    starts.push(total);
    for &count in fold_count {
        total = total.checked_add(count)?;
        starts.push(total);
    }
    Some(starts)
}

fn is_stratified_cv(param: &SvmParameter, nr_fold: usize, l: usize) -> bool {
    matches!(param.svm_type, SvmType::CSvc | SvmType::NuSvc) && nr_fold < l
}

fn evaluate_fold(
    prob: &SvmProblem,
    param: &SvmParameter,
    perm: &[usize],
    begin: usize,
    end: usize,
) -> Option<Vec<f64>> {
    let l = prob.labels.len();

    // Build sub-problem excluding held-out [begin..end)
    if prob.instances.len() != l || perm.len() != l {
        return None;
    }
    let held_len = end.checked_sub(begin)?;
    let sub_l = l.checked_sub(held_len)?;
    let held = perm.get(begin..end)?;
    let mut sub_labels = Vec::with_capacity(sub_l);
    let mut sub_instances = Vec::with_capacity(sub_l);

    for &pi in perm.get(..begin)? {
        sub_labels.push(*prob.labels.get(pi)?);
        sub_instances.push(prob.instances.get(pi)?.clone());
    }
    for &pi in perm.get(end..l)? {
        sub_labels.push(*prob.labels.get(pi)?);
        sub_instances.push(prob.instances.get(pi)?.clone());
    }

    let subprob = SvmProblem {
        labels: sub_labels,
        instances: sub_instances,
    };

    #[cfg(feature = "rayon")]
    let submodel = crate::with_suppressed_info(|| svm_train(&subprob, param));
    #[cfg(not(feature = "rayon"))]
    let submodel = svm_train(&subprob, param);

    held.iter()
        .map(|&i| Some(predict_cv_target(&submodel, param, prob.instances.get(i)?)))
        .collect()
}

// ─── Public API ──────────────────────────────────────────────────────

/// Perform k-fold cross-validation on an SVM problem.
///
/// Returns a `Vec<f64>` of length `prob.labels.len()` where `target[i]`
/// is the prediction for instance `i` when it was held out.
///
/// - **Classification** (C-SVC, ν-SVC) with `nr_fold < l`: stratified
///   splitting that preserves class ratios across folds.
/// - **Regression / one-class** or `nr_fold == l`: simple random split.
///
/// If `nr_fold > l`, clamps to `l` (leave-one-out).
///
/// ```
/// use libsvm_rs::cross_validation::svm_cross_validation;
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
///
/// let targets = svm_cross_validation(&problem, &param, 2);
/// assert_eq!(targets.len(), problem.labels.len());
/// # Ok::<(), libsvm_rs::SvmError>(())
/// ```
#[must_use]
/// # Panics
///
/// Panics if row lengths differ, row counts exceed 2^53, fold arithmetic overflows,
/// internal offsets are inconsistent, or training rejects a fold.
pub fn svm_cross_validation(prob: &SvmProblem, param: &SvmParameter, nr_fold: usize) -> Vec<f64> {
    let targets = try_cross_validation(prob, param, nr_fold);
    #[expect(
        clippy::expect_used,
        reason = "try_cross_validation checks problem lengths, fold arithmetic and permutation slices"
    )]
    targets.expect(
        "cross-validation requires consistent row lengths, fold offsets and at most 2^53 rows",
    )
}

fn try_cross_validation(
    prob: &SvmProblem,
    param: &SvmParameter,
    mut nr_fold: usize,
) -> Option<Vec<f64>> {
    let l = prob.labels.len();
    if prob.instances.len() != l || l as u128 > (1u128 << 53) {
        return None;
    }
    if l == 0 {
        return Some(Vec::new());
    }
    if nr_fold == 0 {
        crate::info(
            "WARNING: # folds (0) <= 0. Will use # folds = # data instead (i.e., leave-one-out cross validation)\n",
        );
        nr_fold = l;
    }
    if nr_fold > l {
        crate::info(&format!(
            "WARNING: # folds ({nr_fold}) > # data ({l}). Will use # folds = # data instead (i.e., leave-one-out cross validation)\n"
        ));
        nr_fold = l;
    }
    let mut perm: Vec<usize> = (0..l).collect();
    let fold_start = if is_stratified_cv(param, nr_fold, l) {
        let grouped = group_classes(&prob.labels)?;
        let mut index = grouped.perm;
        for (&start, &count) in grouped.start.iter().zip(&grouped.count) {
            shuffle_range(&mut index, start, count)?;
        }
        let fold_count: Vec<usize> = (0..nr_fold)
            .map(|i| {
                grouped.count.iter().try_fold(0usize, |total, &count| {
                    let begin = i.checked_mul(count)?.checked_div(nr_fold)?;
                    let end = i.checked_add(1)?.checked_mul(count)?.checked_div(nr_fold)?;
                    total.checked_add(end.checked_sub(begin)?)
                })
            })
            .collect::<Option<_>>()?;
        let starts = fold_starts(&fold_count)?;
        let mut offsets = vec![0usize; nr_fold];
        for (&start, &count) in grouped.start.iter().zip(&grouped.count) {
            for (i, offset) in offsets.iter_mut().enumerate() {
                let begin = start.checked_add(i.checked_mul(count)?.checked_div(nr_fold)?)?;
                let end = start
                    .checked_add(i.checked_add(1)?.checked_mul(count)?.checked_div(nr_fold)?)?;
                let input = index.get(begin..end)?;
                let out_begin = starts.get(i)?.checked_add(*offset)?;
                let out_end = out_begin.checked_add(input.len())?;
                let output = perm.get_mut(out_begin..out_end)?;
                for (slot, &idx) in output.iter_mut().zip(input) {
                    *slot = idx;
                }
                *offset = offset.checked_add(input.len())?;
            }
        }
        starts
    } else {
        shuffle_range(&mut perm, 0, l)?;
        let counts: Vec<usize> = (0..nr_fold)
            .map(|i| {
                let begin = i.checked_mul(l)?.checked_div(nr_fold)?;
                let end = i.checked_add(1)?.checked_mul(l)?.checked_div(nr_fold)?;
                end.checked_sub(begin)
            })
            .collect::<Option<_>>()?;
        fold_starts(&counts)?
    };
    let mut target = vec![0.0; l];
    #[cfg(feature = "rayon")]
    if !param.probability {
        let predictions: Vec<Vec<f64>> = fold_start
            .par_windows(2)
            .map(|bounds| {
                let [begin, end] = bounds else {
                    return None;
                };
                evaluate_fold(prob, param, &perm, *begin, *end)
            })
            .collect::<Option<_>>()?;
        let mut output = perm.iter();
        for fold in predictions {
            for prediction in fold {
                *target.get_mut(*output.next()?)? = prediction;
            }
        }
        return Some(target);
    }
    // Probability training consumes the shared PRNG in fold order.
    for bounds in fold_start.windows(2) {
        let [begin, end] = bounds else {
            return None;
        };
        let predictions = evaluate_fold(prob, param, &perm, *begin, *end)?;
        for (&idx, prediction) in perm.get(*begin..*end)?.iter().zip(predictions) {
            *target.get_mut(idx)? = prediction;
        }
    }
    Some(target)
}

// ─── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::load_problem;
    use crate::types::{KernelType, SvmModel, SvmNode};
    use std::path::PathBuf;

    #[test]
    fn fold_starts_rejects_overflow() {
        assert!(fold_starts(&[usize::MAX, 1]).is_none());
        assert_eq!(fold_starts(&[2, 0, 1]), Some(vec![0, 2, 2, 3]));
    }

    #[test]
    fn evaluate_fold_rejects_inconsistent_rows_and_offsets() {
        let mut problem = SvmProblem {
            labels: vec![1.0],
            instances: Vec::new(),
        };
        let param = SvmParameter::default();
        assert!(evaluate_fold(&problem, &param, &[0], 0, 1).is_none());
        problem.instances.push(Vec::new());
        assert!(evaluate_fold(&problem, &param, &[0], 1, 0).is_none());
        assert!(evaluate_fold(&problem, &param, &[0], 0, 2).is_none());
    }

    #[test]
    fn try_cross_validation_rejects_inconsistent_rows() {
        let problem = SvmProblem {
            labels: vec![1.0],
            instances: Vec::new(),
        };
        assert!(try_cross_validation(&problem, &SvmParameter::default(), 2).is_none());
    }

    fn data_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("data")
    }

    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "Predicted labels are the exact class values 1 and -1"
    )]
    fn cross_validation_basic() {
        let labels = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let instances: Vec<Vec<SvmNode>> = (0..10)
            .map(|i| {
                vec![SvmNode {
                    index: 1,
                    value: f64::from(i) * 0.1,
                }]
            })
            .collect();

        let prob = SvmProblem { labels, instances };
        let param = SvmParameter {
            kernel_type: KernelType::Linear,
            ..Default::default()
        };

        let target = svm_cross_validation(&prob, &param, 5);
        assert_eq!(target.len(), 10);
        for &pred in &target {
            assert!(pred == 1.0 || pred == -1.0);
        }
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
    fn cross_validation_classification() {
        let problem = load_problem(&data_dir().join("heart_scale")).unwrap();
        let param = SvmParameter {
            svm_type: SvmType::CSvc,
            kernel_type: KernelType::Rbf,
            gamma: 1.0 / 13.0,
            c: 1.0,
            cache_size: 100.0,
            eps: 0.001,
            shrinking: true,
            ..Default::default()
        };

        let target = svm_cross_validation(&problem, &param, 5);
        assert_eq!(target.len(), problem.labels.len());

        let correct = target
            .iter()
            .zip(problem.labels.iter())
            .filter(|&(&pred, &label)| pred == label)
            .count();
        let accuracy = correct as f64 / problem.labels.len() as f64;
        assert!(
            accuracy > 0.70,
            "5-fold CV accuracy {:.1}% too low (expected >70%)",
            accuracy * 100.0
        );
    }

    #[test]
    #[expect(
        clippy::cast_precision_loss,
        reason = "C converts the same integer counts to double at these points"
    )]
    fn cross_validation_regression() {
        let problem = load_problem(&data_dir().join("housing_scale")).unwrap();
        let param = SvmParameter {
            svm_type: SvmType::EpsilonSvr,
            kernel_type: KernelType::Rbf,
            gamma: 1.0 / 13.0,
            c: 1.0,
            p: 0.1,
            cache_size: 100.0,
            eps: 0.001,
            shrinking: true,
            ..Default::default()
        };

        let target = svm_cross_validation(&problem, &param, 5);
        assert_eq!(target.len(), problem.labels.len());

        let mse: f64 = target
            .iter()
            .zip(problem.labels.iter())
            .map(|(&pred, &label)| (pred - label).powi(2))
            .sum::<f64>()
            / problem.labels.len() as f64;
        assert!(mse.is_finite(), "MSE is not finite");
        assert!(mse < 500.0, "MSE {mse} too high");
    }

    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "Predicted labels are the exact class values 1 and -1"
    )]
    fn cross_validation_zero_folds_clamps_to_leave_one_out() {
        let labels = vec![1.0, -1.0, 1.0, -1.0];
        let instances: Vec<Vec<SvmNode>> = vec![
            vec![SvmNode {
                index: 1,
                value: 1.0,
            }],
            vec![SvmNode {
                index: 1,
                value: -1.0,
            }],
            vec![SvmNode {
                index: 1,
                value: 0.8,
            }],
            vec![SvmNode {
                index: 1,
                value: -0.9,
            }],
        ];
        let prob = SvmProblem { labels, instances };
        let param = SvmParameter {
            svm_type: SvmType::CSvc,
            kernel_type: KernelType::Linear,
            c: 1.0,
            eps: 0.001,
            ..Default::default()
        };

        let target = svm_cross_validation(&prob, &param, 0);
        assert_eq!(target.len(), prob.labels.len());
        for &pred in &target {
            assert!(pred == 1.0 || pred == -1.0);
        }
    }

    #[test]
    fn cross_validation_empty_problem_returns_empty() {
        let prob = SvmProblem {
            labels: Vec::new(),
            instances: Vec::new(),
        };
        let target = svm_cross_validation(&prob, &SvmParameter::default(), 5);
        assert_eq!(target.len(), 0);
    }

    #[test]
    fn predict_cv_target_uses_probability_label_for_classification() {
        let param = SvmParameter {
            svm_type: SvmType::CSvc,
            kernel_type: KernelType::Linear,
            probability: true,
            ..Default::default()
        };
        let model = SvmModel {
            param: param.clone(),
            nr_class: 2,
            sv: vec![
                vec![SvmNode {
                    index: 1,
                    value: 1.0,
                }],
                vec![SvmNode {
                    index: 1,
                    value: 1.0,
                }],
            ],
            sv_coef: vec![vec![1.0, -1.0]],
            rho: vec![-1.0],
            prob_a: vec![1.0],
            prob_b: vec![0.0],
            prob_density_marks: Vec::new(),
            sv_indices: vec![1, 2],
            label: vec![1, -1],
            n_sv: vec![1, 1],
        };
        let x = vec![SvmNode {
            index: 1,
            value: 1.0,
        }];

        let vote_label = predict(&model, &x);
        let (prob_label, _) = predict_probability(&model, &x).unwrap();

        assert_eq!(vote_label, 1.0);
        assert_eq!(prob_label, -1.0);
        assert_eq!(predict_cv_target(&model, &param, &x), prob_label);
    }
}
