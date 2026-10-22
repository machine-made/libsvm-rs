//! Probability estimation functions for SVM models.
//!
//! Provides Platt scaling (`sigmoid_train`/`sigmoid_predict`), multiclass
//! probability estimation, and density-based probability for one-class SVM.
//! Matches the original LIBSVM's probability routines (svm.cpp:1714–2096).

use crate::predict::predict_values;
use crate::train::svm_train;
use crate::types::{SvmModel, SvmParameter, SvmProblem};
use crate::util::c_rand;
#[cfg(feature = "rayon")]
use rayon::prelude::*;

// ─── Platt scaling ───────────────────────────────────────────────────

/// Train Platt scaling parameters (A, B) via Newton's method.
///
/// Given decision values and labels (+1/−1), fits the sigmoid
/// P(y=1|f) = 1/(1+exp(A*f+B)) using the algorithm of Lin, Lin &
/// Weng (2007). Matches LIBSVM's `sigmoid_train`.
pub fn sigmoid_train(dec_values: &[f64], labels: &[f64]) -> (f64, f64) {
    let l = dec_values.len();

    let mut prior1: f64 = 0.0;
    let mut prior0: f64 = 0.0;
    for &y in labels {
        if y > 0.0 {
            prior1 += 1.0;
        } else {
            prior0 += 1.0;
        }
    }

    let max_iter = 100;
    let min_step = 1e-10;
    let sigma = 1e-12;
    let eps = 1e-5;

    let hi_target = (prior1 + 1.0) / (prior1 + 2.0);
    let lo_target = 1.0 / (prior0 + 2.0);

    let t: Vec<f64> = labels
        .iter()
        .map(|&y| if y > 0.0 { hi_target } else { lo_target })
        .collect();

    // Initial point
    let mut a = 0.0;
    let mut b = ((prior0 + 1.0) / (prior1 + 1.0)).ln();

    // Initial objective
    let mut fval = 0.0;
    for i in 0..l {
        let f_apb = dec_values[i] * a + b;
        if f_apb >= 0.0 {
            fval += t[i] * f_apb + (1.0 + (-f_apb).exp()).ln();
        } else {
            fval += (t[i] - 1.0) * f_apb + (1.0 + f_apb.exp()).ln();
        }
    }

    for _iter in 0..max_iter {
        // Gradient and Hessian (H' = H + σI)
        let mut h11 = sigma;
        let mut h22 = sigma;
        let mut h21 = 0.0;
        let mut g1 = 0.0;
        let mut g2 = 0.0;

        for i in 0..l {
            let f_apb = dec_values[i] * a + b;
            let (p, q) = if f_apb >= 0.0 {
                let e = (-f_apb).exp();
                (e / (1.0 + e), 1.0 / (1.0 + e))
            } else {
                let e = f_apb.exp();
                (1.0 / (1.0 + e), e / (1.0 + e))
            };
            let d2 = p * q;
            h11 += dec_values[i] * dec_values[i] * d2;
            h22 += d2;
            h21 += dec_values[i] * d2;
            let d1 = t[i] - p;
            g1 += dec_values[i] * d1;
            g2 += d1;
        }

        if g1.abs() < eps && g2.abs() < eps {
            break;
        }

        // Newton direction: −H'⁻¹ g
        let det = h11 * h22 - h21 * h21;
        let da = -(h22 * g1 - h21 * g2) / det;
        let db = -(-h21 * g1 + h11 * g2) / det;
        let gd = g1 * da + g2 * db;

        // Line search with step-size halving
        let mut stepsize = 1.0;
        while stepsize >= min_step {
            let new_a = a + stepsize * da;
            let new_b = b + stepsize * db;

            let mut newf = 0.0;
            for i in 0..l {
                let f_apb = dec_values[i] * new_a + new_b;
                if f_apb >= 0.0 {
                    newf += t[i] * f_apb + (1.0 + (-f_apb).exp()).ln();
                } else {
                    newf += (t[i] - 1.0) * f_apb + (1.0 + f_apb.exp()).ln();
                }
            }

            if newf < fval + 0.0001 * stepsize * gd {
                a = new_a;
                b = new_b;
                fval = newf;
                break;
            }
            stepsize /= 2.0;
        }

        if stepsize < min_step {
            break;
        }
    }

    (a, b)
}

/// Numerically stable sigmoid prediction.
///
/// Returns P(y=1|f) = 1/(1+exp(A*f+B)), branching on sign of A*f+B
/// to avoid overflow. Matches LIBSVM's `sigmoid_predict`.
pub fn sigmoid_predict(decision_value: f64, a: f64, b: f64) -> f64 {
    let f_apb = decision_value * a + b;
    if f_apb >= 0.0 {
        (-f_apb).exp() / (1.0 + (-f_apb).exp())
    } else {
        1.0 / (1.0 + f_apb.exp())
    }
}

// ─── Multiclass probability ──────────────────────────────────────────

/// Solve multiclass probabilities from pairwise estimates.
///
/// Given `k` classes and a k×k matrix `r` of pairwise probabilities
/// (r\[i\]\[j\] = P(class i | class i or j)), fills `p` with class
/// probabilities using the Wu-Lin-Weng iterative method.
///
/// Matches LIBSVM's `multiclass_probability`.
#[allow(clippy::needless_range_loop)]
pub fn multiclass_probability(k: usize, r: &[Vec<f64>], p: &mut [f64]) {
    let max_iter = 100.max(k);
    let eps = 0.005 / k as f64;

    // Build Q matrix
    let mut q_mat = vec![vec![0.0; k]; k];
    for t in 0..k {
        q_mat[t][t] = 0.0;
        for j in 0..t {
            q_mat[t][t] += r[j][t] * r[j][t];
            q_mat[t][j] = q_mat[j][t];
        }
        for j in (t + 1)..k {
            q_mat[t][t] += r[j][t] * r[j][t];
            q_mat[t][j] = -r[j][t] * r[t][j];
        }
    }

    for t in 0..k {
        p[t] = 1.0 / k as f64;
    }

    let mut qp = vec![0.0; k];

    for _iter in 0..max_iter {
        let mut p_qp = 0.0;
        for t in 0..k {
            qp[t] = 0.0;
            for j in 0..k {
                qp[t] += q_mat[t][j] * p[j];
            }
            p_qp += p[t] * qp[t];
        }

        let mut max_error = 0.0;
        for t in 0..k {
            let error = (qp[t] - p_qp).abs();
            if error > max_error {
                max_error = error;
            }
        }
        if max_error < eps {
            break;
        }

        for t in 0..k {
            let diff = (-qp[t] + p_qp) / q_mat[t][t];
            p[t] += diff;
            p_qp = (p_qp + diff * (diff * q_mat[t][t] + 2.0 * qp[t])) / (1.0 + diff) / (1.0 + diff);
            for j in 0..k {
                qp[j] = (qp[j] + diff * q_mat[t][j]) / (1.0 + diff);
                p[j] /= 1.0 + diff;
            }
        }
    }
}

// ─── Binary SVC probability via internal CV ──────────────────────────

fn evaluate_binary_svc_probability_fold(
    prob: &SvmProblem,
    param: &SvmParameter,
    cp: f64,
    cn: f64,
    perm: &[usize],
    begin: usize,
    end: usize,
) -> Vec<f64> {
    let l = prob.labels.len();

    // Build training sub-problem (exclude held-out fold)
    let mut sub_instances = Vec::with_capacity(l - (end - begin));
    let mut sub_labels = Vec::with_capacity(l - (end - begin));

    for &pi in &perm[..begin] {
        sub_instances.push(prob.instances[pi].clone());
        sub_labels.push(prob.labels[pi]);
    }
    for &pi in &perm[end..l] {
        sub_instances.push(prob.instances[pi].clone());
        sub_labels.push(prob.labels[pi]);
    }

    // Count classes in training set
    let p_count = sub_labels.iter().filter(|&&y| y > 0.0).count();
    let n_count = sub_labels.len() - p_count;

    if p_count == 0 && n_count == 0 {
        vec![0.0; end - begin]
    } else if p_count > 0 && n_count == 0 {
        vec![1.0; end - begin]
    } else if p_count == 0 && n_count > 0 {
        vec![-1.0; end - begin]
    } else {
        let mut subparam = param.clone();
        subparam.probability = false;
        subparam.c = 1.0;
        subparam.weight = vec![(1, cp), (-1, cn)];

        let subprob = SvmProblem {
            labels: sub_labels,
            instances: sub_instances,
        };
        #[cfg(feature = "rayon")]
        let submodel = crate::with_suppressed_info(|| svm_train(&subprob, &subparam));
        #[cfg(not(feature = "rayon"))]
        let submodel = svm_train(&subprob, &subparam);

        (begin..end)
            .map(|j| {
                let mut dv = [0.0];
                predict_values(&submodel, &prob.instances[perm[j]], &mut dv);
                // Sign correction: ensure +1/−1 ordering
                dv[0] * submodel.label[0] as f64
            })
            .collect()
    }
}

/// Estimate Platt scaling parameters for a binary sub-problem.
///
/// Performs 5-fold CV internally: trains on 4 folds with class weights
/// (cp, cn), collects decision values on the held-out fold, then fits
/// a sigmoid via `sigmoid_train`.
///
/// Matches LIBSVM's `svm_binary_svc_probability`.
pub fn svm_binary_svc_probability(
    prob: &SvmProblem,
    param: &SvmParameter,
    cp: f64,
    cn: f64,
) -> (f64, f64) {
    let l = prob.labels.len();
    let nr_fold = 5.min(l.max(1));
    let mut perm: Vec<usize> = (0..l).collect();
    let mut dec_values = vec![0.0; l];

    // Random shuffle (Fisher-Yates)
    for i in 0..l {
        let j = i + c_rand() % (l - i);
        perm.swap(i, j);
    }

    #[cfg(feature = "rayon")]
    {
        let fold_predictions: Vec<Vec<f64>> = (0..nr_fold)
            .into_par_iter()
            .map(|fold| {
                let begin = fold * l / nr_fold;
                let end = (fold + 1) * l / nr_fold;
                evaluate_binary_svc_probability_fold(prob, param, cp, cn, &perm, begin, end)
            })
            .collect();

        for (fold, predictions) in fold_predictions.into_iter().enumerate() {
            let begin = fold * l / nr_fold;
            let end = (fold + 1) * l / nr_fold;
            for (j, prediction) in (begin..end).zip(predictions.into_iter()) {
                dec_values[perm[j]] = prediction;
            }
        }
    }

    #[cfg(not(feature = "rayon"))]
    {
        for fold in 0..nr_fold {
            let begin = fold * l / nr_fold;
            let end = (fold + 1) * l / nr_fold;
            let predictions =
                evaluate_binary_svc_probability_fold(prob, param, cp, cn, &perm, begin, end);

            for (j, prediction) in (begin..end).zip(predictions.into_iter()) {
                dec_values[perm[j]] = prediction;
            }
        }
    }

    sigmoid_train(&dec_values, &prob.labels)
}

// ─── One-class probability ───────────────────────────────────────────

/// Predict probability for one-class SVM from density marks.
///
/// Bin-lookup in precomputed density marks (10 entries). Returns a
/// probability estimate in (0, 1).
///
/// Matches LIBSVM's `predict_one_class_probability`.
pub fn predict_one_class_probability(prob_density_marks: &[f64], dec_value: f64) -> f64 {
    let nr_marks = prob_density_marks.len();
    if nr_marks == 0 {
        return 0.5;
    }

    if dec_value < prob_density_marks[0] {
        return 0.001;
    }
    if dec_value > prob_density_marks[nr_marks - 1] {
        return 0.999;
    }

    for (i, &mark) in prob_density_marks
        .iter()
        .enumerate()
        .skip(1)
        .take(nr_marks - 1)
    {
        if dec_value < mark {
            return i as f64 / nr_marks as f64;
        }
    }

    0.999
}

/// Estimate probability density marks for one-class SVM.
///
/// Predicts all training instances, sorts decision values, bins into
/// 10 density marks. Returns `None` if fewer than 5 positive or 5
/// negative decision values.
///
/// Matches LIBSVM's `svm_one_class_probability`.
pub fn svm_one_class_probability(prob: &SvmProblem, model: &SvmModel) -> Option<Vec<f64>> {
    let l = prob.labels.len();
    let mut dec_values = vec![0.0; l];

    for (dv_slot, instance) in dec_values.iter_mut().zip(prob.instances.iter()) {
        let mut dv = [0.0];
        predict_values(model, instance, &mut dv);
        *dv_slot = dv[0];
    }

    dec_values.sort_by(f64::total_cmp);

    // Find first index with dec_value >= 0  (= neg_counter in C++)
    let mut neg_counter = 0usize;
    for (i, &dv) in dec_values.iter().enumerate() {
        if dv >= 0.0 {
            neg_counter = i;
            break;
        }
    }
    let pos_counter = l - neg_counter;

    let nr_marks: usize = 10;
    let mid = nr_marks / 2; // 5

    if neg_counter < mid || pos_counter < mid {
        crate::info(&format!(
            "WARNING: number of positive or negative decision values <{}; \
             too few to do a probability estimation.\n",
            mid
        ));
        return None;
    }

    let mut tmp_marks = vec![0.0; nr_marks + 1];

    for i in 0..mid {
        tmp_marks[i] = dec_values[i * neg_counter / mid];
    }
    tmp_marks[mid] = 0.0;
    for i in (mid + 1)..=nr_marks {
        tmp_marks[i] = dec_values[neg_counter - 1 + (i - mid) * pos_counter / mid];
    }

    let mut marks = vec![0.0; nr_marks];
    for i in 0..nr_marks {
        marks[i] = (tmp_marks[i] + tmp_marks[i + 1]) / 2.0;
    }

    Some(marks)
}

// ─── SVR probability ─────────────────────────────────────────────────

/// Estimate Laplace scale parameter for SVR probability.
///
/// Performs 5-fold CV to get residuals, computes MAE, then applies
/// outlier rejection (exclude |residual| > 5·√(2·mae²)) and
/// recomputes. Returns the final MAE (= σ of Laplace distribution).
///
/// Matches LIBSVM's `svm_svr_probability`.
pub fn svm_svr_probability(prob: &SvmProblem, param: &SvmParameter) -> f64 {
    let l = prob.labels.len();
    let nr_fold = 5;

    let mut newparam = param.clone();
    newparam.probability = false;
    let ymv = crate::cross_validation::svm_cross_validation(prob, &newparam, nr_fold);

    // Compute residuals and initial MAE
    let mut ymv_residuals: Vec<f64> = Vec::with_capacity(l);
    let mut mae = 0.0;
    for (&label, &pred) in prob.labels.iter().zip(ymv.iter()) {
        let r = label - pred;
        ymv_residuals.push(r);
        mae += r.abs();
    }
    mae /= l as f64;

    // Outlier rejection
    let std_val = (2.0 * mae * mae).sqrt();
    let mut count = 0usize;
    mae = 0.0;
    for &residual in &ymv_residuals {
        if residual.abs() > 5.0 * std_val {
            count += 1;
        } else {
            mae += residual.abs();
        }
    }
    mae /= (l - count) as f64;

    crate::info(&format!(
        "Prob. model for test data: target value = predicted value + z,\n\
         z: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma= {:.6}\n",
        mae
    ));

    mae
}

// ─── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::load_problem;
    use crate::train::svm_train;
    use crate::types::{KernelType, SvmNode, SvmParameter, SvmProblem, SvmType};
    use std::path::PathBuf;

    fn data_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("data")
    }

    #[test]
    fn sigmoid_predict_symmetric() {
        let p = sigmoid_predict(0.0, 0.0, 0.0);
        assert!((p - 0.5).abs() < 1e-10);
    }

    #[test]
    fn sigmoid_predict_stable() {
        let p1 = sigmoid_predict(1000.0, 1.0, 0.0);
        assert!(p1.is_finite() && (0.0..=1.0).contains(&p1));

        let p2 = sigmoid_predict(-1000.0, 1.0, 0.0);
        assert!(p2.is_finite() && (0.0..=1.0).contains(&p2));
    }

    #[test]
    fn sigmoid_train_basic() {
        let dec = vec![1.0, 2.0, -1.0, -2.0, 0.5];
        let lab = vec![1.0, 1.0, -1.0, -1.0, 1.0];
        let (a, b) = sigmoid_train(&dec, &lab);
        assert!(a.is_finite());
        assert!(b.is_finite());
    }

    #[test]
    fn multiclass_prob_sums_to_one() {
        let k = 3;
        let r = vec![
            vec![0.0, 0.6, 0.5],
            vec![0.4, 0.0, 0.7],
            vec![0.5, 0.3, 0.0],
        ];
        let mut p = vec![0.0; k];
        multiclass_probability(k, &r, &mut p);

        let sum: f64 = p.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "probabilities sum to {}, expected ~1.0",
            sum
        );
        for &pi in &p {
            assert!(pi > 0.0, "probability should be positive, got {}", pi);
        }
    }

    #[test]
    fn predict_one_class_prob_boundaries() {
        let marks = vec![-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9];
        assert!((predict_one_class_probability(&marks, -1.0) - 0.001).abs() < 1e-10);
        assert!((predict_one_class_probability(&marks, 1.0) - 0.999).abs() < 1e-10);
        let mid = predict_one_class_probability(&marks, 0.0);
        assert!(mid > 0.0 && mid < 1.0);
    }

    #[test]
    fn predict_one_class_probability_empty_marks_is_half() {
        let p = predict_one_class_probability(&[], 0.25);
        assert!((p - 0.5).abs() < 1e-12);
    }

    #[test]
    fn binary_svc_probability_is_finite() {
        let prob = load_problem(&data_dir().join("heart_scale")).unwrap();
        let param = SvmParameter {
            svm_type: SvmType::CSvc,
            kernel_type: KernelType::Rbf,
            gamma: 1.0 / 13.0,
            ..Default::default()
        };

        let (a, b) = svm_binary_svc_probability(&prob, &param, 1.0, 1.0);
        assert!(a.is_finite());
        assert!(b.is_finite());
    }

    #[test]
    fn one_class_probability_marks_generated_for_large_problem() {
        let prob = load_problem(&data_dir().join("heart_scale")).unwrap();
        let param = SvmParameter {
            svm_type: SvmType::OneClass,
            kernel_type: KernelType::Rbf,
            gamma: 1.0 / 13.0,
            nu: 0.5,
            ..Default::default()
        };
        let model = svm_train(&prob, &param);

        let marks = svm_one_class_probability(&prob, &model).expect("expected marks");
        assert_eq!(marks.len(), 10);
        for pair in marks.windows(2) {
            assert!(pair[0] <= pair[1]);
        }
    }

    #[test]
    fn one_class_probability_returns_none_when_too_few_samples() {
        let prob = SvmProblem {
            labels: vec![1.0, 1.0, 1.0, 1.0],
            instances: vec![
                vec![SvmNode {
                    index: 1,
                    value: 0.0,
                }],
                vec![SvmNode {
                    index: 1,
                    value: 1.0,
                }],
                vec![SvmNode {
                    index: 1,
                    value: 2.0,
                }],
                vec![SvmNode {
                    index: 1,
                    value: 3.0,
                }],
            ],
        };
        let param = SvmParameter {
            svm_type: SvmType::OneClass,
            kernel_type: KernelType::Linear,
            nu: 0.5,
            ..Default::default()
        };
        let model = svm_train(&prob, &param);
        assert!(svm_one_class_probability(&prob, &model).is_none());
    }

    #[test]
    fn svr_probability_is_positive_and_finite() {
        let prob = load_problem(&data_dir().join("housing_scale")).unwrap();
        let param = SvmParameter {
            svm_type: SvmType::EpsilonSvr,
            kernel_type: KernelType::Rbf,
            gamma: 1.0 / 13.0,
            c: 1.0,
            p: 0.1,
            ..Default::default()
        };

        let sigma = svm_svr_probability(&prob, &param);
        assert!(sigma.is_finite());
        assert!(sigma > 0.0);
    }
}
