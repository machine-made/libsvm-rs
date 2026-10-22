//! Cross-validation for SVM models.
//!
//! Provides stratified cross-validation for classification and simple
//! random-split cross-validation for regression/one-class problems.
//! Matches LIBSVM's `svm_cross_validation` (svm.cpp:2437–2556).

use crate::predict::{predict, predict_probability};
use crate::train::svm_train;
use crate::types::{SvmModel, SvmNode, SvmParameter, SvmProblem, SvmType};
use crate::util::{group_classes, shuffle_range};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

fn predict_cv_target(model: &SvmModel, param: &SvmParameter, x: &[SvmNode]) -> f64 {
    if param.probability && matches!(param.svm_type, SvmType::CSvc | SvmType::NuSvc) {
        predict_probability(model, x)
            .map(|(label, _)| label)
            .unwrap_or_else(|| predict(model, x))
    } else {
        predict(model, x)
    }
}

fn fold_starts(fold_count: &[usize]) -> Vec<usize> {
    let mut fold_start = vec![0usize; fold_count.len() + 1];
    for i in 0..fold_count.len() {
        fold_start[i + 1] = fold_start[i] + fold_count[i];
    }
    fold_start
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
) -> Vec<f64> {
    let l = prob.labels.len();

    // Build sub-problem excluding held-out [begin..end)
    let sub_l = l - (end - begin);
    let mut sub_labels = Vec::with_capacity(sub_l);
    let mut sub_instances = Vec::with_capacity(sub_l);

    for &pi in &perm[..begin] {
        sub_labels.push(prob.labels[pi]);
        sub_instances.push(prob.instances[pi].clone());
    }
    for &pi in &perm[end..l] {
        sub_labels.push(prob.labels[pi]);
        sub_instances.push(prob.instances[pi].clone());
    }

    let subprob = SvmProblem {
        labels: sub_labels,
        instances: sub_instances,
    };

    #[cfg(feature = "rayon")]
    let submodel = crate::with_suppressed_info(|| svm_train(&subprob, param));
    #[cfg(not(feature = "rayon"))]
    let submodel = svm_train(&subprob, param);

    (begin..end)
        .map(|j| predict_cv_target(&submodel, param, &prob.instances[perm[j]]))
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
pub fn svm_cross_validation(
    prob: &SvmProblem,
    param: &SvmParameter,
    mut nr_fold: usize,
) -> Vec<f64> {
    let l = prob.labels.len();

    if l == 0 {
        return Vec::new();
    }

    if nr_fold == 0 {
        crate::info(
            "WARNING: # folds (0) <= 0. Will use # folds = # data instead \
             (i.e., leave-one-out cross validation)\n",
        );
        nr_fold = l;
    }

    if nr_fold > l {
        crate::info(&format!(
            "WARNING: # folds ({}) > # data ({}). Will use # folds = # data instead \
             (i.e., leave-one-out cross validation)\n",
            nr_fold, l
        ));
        nr_fold = l;
    }

    let mut perm: Vec<usize> = (0..l).collect();

    // ── Fold assignment ──────────────────────────────────────────
    let (_fold_count, fold_start) = if is_stratified_cv(param, nr_fold, l) {
        // Stratified: group by class, shuffle within class, distribute
        let grouped = group_classes(&prob.labels);
        let nr_class = grouped.label.len();
        let start = &grouped.start;
        let count = &grouped.count;
        let mut index = grouped.perm;

        // Shuffle within each class
        for c in 0..nr_class {
            let s = start[c];
            let n = count[c];
            shuffle_range(&mut index, s, n);
        }

        // Compute fold sizes
        let fold_count: Vec<usize> = (0..nr_fold)
            .map(|i| {
                count
                    .iter()
                    .map(|&cnt| ((i + 1) * cnt) / nr_fold - (i * cnt) / nr_fold)
                    .sum()
            })
            .collect();
        let fold_start = fold_starts(&fold_count);
        // Distribute samples to folds, preserving class balance
        // (C++ increments fold_start[i] as a running pointer; we
        // use a separate offset array to keep fold_start immutable.)
        let mut offset = vec![0usize; nr_fold];
        for c in 0..nr_class {
            for i in 0..nr_fold {
                let begin = start[c] + (i * count[c]) / nr_fold;
                let end = start[c] + ((i + 1) * count[c]) / nr_fold;
                for &idx in &index[begin..end] {
                    perm[fold_start[i] + offset[i]] = idx;
                    offset[i] += 1;
                }
            }
        }
        (fold_count, fold_start)
    } else {
        // Simple random shuffle
        shuffle_range(&mut perm, 0, l);
        let fold_count: Vec<usize> = (0..nr_fold)
            .map(|i| (i + 1) * l / nr_fold - i * l / nr_fold)
            .collect();
        let fold_start = fold_starts(&fold_count);
        (fold_count, fold_start)
    };

    // ── Evaluate each fold ───────────────────────────────────────
    let mut target = vec![0.0; l];

    #[cfg(feature = "rayon")]
    {
        if param.probability {
            // Probability training performs its own deterministic CV shuffles.
            // Keep this outer loop serial so the shared LIBSVM-compatible PRNG
            // is consumed in the same order as the default build.
            for i in 0..nr_fold {
                let begin = fold_start[i];
                let end = fold_start[i + 1];
                let predictions = evaluate_fold(prob, param, &perm, begin, end);

                for (j, prediction) in (begin..end).zip(predictions.into_iter()) {
                    target[perm[j]] = prediction;
                }
            }
        } else {
            let mut target_in_permuted_order = vec![0.0; l];
            let mut fold_slices = Vec::with_capacity(nr_fold);
            let mut remaining = target_in_permuted_order.as_mut_slice();
            for i in 0..nr_fold {
                let fold_len = fold_start[i + 1] - fold_start[i];
                let (fold_slice, rest) = remaining.split_at_mut(fold_len);
                fold_slices.push(fold_slice);
                remaining = rest;
            }

            fold_slices
                .into_par_iter()
                .enumerate()
                .for_each(|(i, fold_target)| {
                    let begin = fold_start[i];
                    let end = fold_start[i + 1];
                    let predictions = evaluate_fold(prob, param, &perm, begin, end);
                    fold_target.copy_from_slice(&predictions);
                });

            for (prediction, &original_idx) in target_in_permuted_order.iter().zip(perm.iter()) {
                target[original_idx] = *prediction;
            }
        }
    }

    #[cfg(not(feature = "rayon"))]
    {
        for i in 0..nr_fold {
            let begin = fold_start[i];
            let end = fold_start[i + 1];
            let predictions = evaluate_fold(prob, param, &perm, begin, end);

            for (j, prediction) in (begin..end).zip(predictions.into_iter()) {
                target[perm[j]] = prediction;
            }
        }
    }

    target
}

// ─── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::load_problem;
    use crate::types::{KernelType, SvmModel, SvmNode};
    use std::path::PathBuf;

    fn data_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("data")
    }

    #[test]
    fn cross_validation_basic() {
        let labels = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let instances: Vec<Vec<SvmNode>> = (0..10)
            .map(|i| {
                vec![SvmNode {
                    index: 1,
                    value: i as f64 * 0.1,
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
            .filter(|(&pred, &label)| pred == label)
            .count();
        let accuracy = correct as f64 / problem.labels.len() as f64;
        assert!(
            accuracy > 0.70,
            "5-fold CV accuracy {:.1}% too low (expected >70%)",
            accuracy * 100.0
        );
    }

    #[test]
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
        assert!(mse < 500.0, "MSE {} too high", mse);
    }

    #[test]
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
        assert!(target.is_empty());
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
