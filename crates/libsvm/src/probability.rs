//! Probability estimation functions for SVM models.
//!
//! Provides Platt scaling (`sigmoid_train`/`sigmoid_predict`), multiclass
//! probability estimation, and density-based probability for one-class SVM.
//! Matches the original LIBSVM's probability routines (svm.cpp:1714–2096).

#![cfg_attr(
    test,
    expect(
        clippy::unwrap_used,
        reason = "Tests unwrap fixture loading and checked assertions"
    )
)]
#![cfg_attr(
    test,
    expect(
        clippy::expect_used,
        reason = "Tests unwrap fixture loading and checked assertions"
    )
)]

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
#[must_use]
/// # Panics
///
/// Panics if lengths differ or the row count exceeds 2^53.
///
/// # Example
///
/// ```
/// use libsvm_rs::probability::{sigmoid_predict, sigmoid_train};
/// let (a, b) = sigmoid_train(&[-2.0, -1.0, 1.0, 2.0], &[-1.0, -1.0, 1.0, 1.0]);
/// assert!(a.is_finite() && b.is_finite());
/// assert!(sigmoid_predict(2.0, a, b) > sigmoid_predict(-2.0, a, b));
/// Ok::<(), libsvm_rs::SvmError>(())
/// ```
pub fn sigmoid_train(dec_values: &[f64], labels: &[f64]) -> (f64, f64) {
    let parameters = try_sigmoid_train(dec_values, labels);
    #[expect(
        clippy::expect_used,
        reason = "try_sigmoid_train checks equal lengths and the exact f64 count limit"
    )]
    parameters.expect("sigmoid training requires equal lengths and at most 2^53 rows; pass one label per decision value")
}

#[expect(
    clippy::many_single_char_names,
    reason = "The C solver uses these short names"
)]
fn try_sigmoid_train(dec_values: &[f64], labels: &[f64]) -> Option<(f64, f64)> {
    if dec_values.len() != labels.len() || dec_values.len() as u128 > (1u128 << 53) {
        return None;
    }

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
    for (&decision, &target) in dec_values.iter().zip(&t) {
        let f_apb = decision * a + b;
        if f_apb >= 0.0 {
            fval += target * f_apb + (1.0 + (-f_apb).exp()).ln();
        } else {
            fval += (target - 1.0) * f_apb + (1.0 + f_apb.exp()).ln();
        }
    }

    for _iter in 0..max_iter {
        // Gradient and Hessian (H' = H + σI)
        let mut h11 = sigma;
        let mut h22 = sigma;
        let mut h21 = 0.0;
        let mut g1 = 0.0;
        let mut g2 = 0.0;

        for (&decision, &target) in dec_values.iter().zip(&t) {
            let f_apb = decision * a + b;
            let (p, q) = if f_apb >= 0.0 {
                let e = (-f_apb).exp();
                (e / (1.0 + e), 1.0 / (1.0 + e))
            } else {
                let e = f_apb.exp();
                (1.0 / (1.0 + e), e / (1.0 + e))
            };
            let d2 = p * q;
            h11 += decision * decision * d2;
            h22 += d2;
            h21 += decision * d2;
            let d1 = target - p;
            g1 += decision * d1;
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
            for (&decision, &target) in dec_values.iter().zip(&t) {
                let f_apb = decision * new_a + new_b;
                if f_apb >= 0.0 {
                    newf += target * f_apb + (1.0 + (-f_apb).exp()).ln();
                } else {
                    newf += (target - 1.0) * f_apb + (1.0 + f_apb.exp()).ln();
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

    Some((a, b))
}

/// Numerically stable sigmoid prediction.
///
/// Returns P(y=1|f) = 1/(1+exp(A*f+B)), branching on sign of A*f+B
/// to avoid overflow. Matches LIBSVM's `sigmoid_predict`.
///
/// # Example
///
/// ```
/// use libsvm_rs::probability::sigmoid_predict;
/// assert_eq!(sigmoid_predict(0.0, -1.0, 0.0).to_bits(), 0.5_f64.to_bits());
/// Ok::<(), libsvm_rs::SvmError>(())
/// ```
#[must_use]
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
/// # Panics
///
/// Panics if the matrix or output is smaller than `k`, or `k` exceeds 2^53.
///
/// # Example
///
/// ```
/// use libsvm_rs::probability::multiclass_probability;
/// let pairwise = vec![vec![0.0, 0.5], vec![0.5, 0.0]];
/// let mut probabilities = [0.0; 2];
/// multiclass_probability(2, &pairwise, &mut probabilities);
/// assert!(probabilities.iter().all(|p| (p - 0.5).abs() < 1e-12));
/// Ok::<(), libsvm_rs::SvmError>(())
/// ```
pub fn multiclass_probability(k: usize, r: &[Vec<f64>], p: &mut [f64]) {
    let result = try_multiclass_probability(k, r, p);
    #[expect(
        clippy::expect_used,
        reason = "try_multiclass_probability checks matrix dimensions, output length and class count"
    )]
    result.expect(
        "multiclass probability requires a k by k matrix, k output slots and at most 2^53 classes. Pass r with k rows of k values and p with k slots",
    );
}

#[expect(
    clippy::cast_precision_loss,
    reason = "The class count is checked against 2^53 before conversion"
)]
fn try_multiclass_probability(k: usize, r: &[Vec<f64>], p: &mut [f64]) -> Option<()> {
    if k as u128 > (1u128 << 53) {
        return None;
    }
    let r: Vec<&[f64]> = r
        .get(..k)?
        .iter()
        .map(|row| row.get(..k))
        .collect::<Option<_>>()?;
    let p = p.get_mut(..k)?;
    let max_iter = 100.max(k);
    let eps = 0.005 / k as f64;
    let mut q_mat = vec![vec![0.0; k]; k];
    for t in 0..k {
        let (before, after) = q_mat.split_at_mut_checked(t)?;
        let row = after.first_mut()?;
        let r_t = r.get(t)?;
        let mut diagonal = 0.0;
        for ((slot, previous), r_j) in row.iter_mut().zip(before).zip(&r) {
            // r rows were checked to have k entries above; t is below k.
            #[expect(
                clippy::indexing_slicing,
                reason = "r.get(..k) and row.get(..k) above establish k entries; t is below k"
            )]
            let rjt = r_j[t];
            diagonal += rjt * rjt;
            // q_mat was allocated with k entries per row; t is below k.
            #[expect(
                clippy::indexing_slicing,
                reason = "q_mat has k entries per row by construction; t is below k"
            )]
            let previous_value = previous[t];
            *slot = previous_value;
        }
        let next = t.checked_add(1)?;
        for ((slot, r_j), &rtj) in row
            .get_mut(next..)?
            .iter_mut()
            .zip(r.get(next..)?)
            .zip(r_t.get(next..)?)
        {
            // r rows were checked to have k entries above; t is below k.
            #[expect(
                clippy::indexing_slicing,
                reason = "r.get(..k) and row.get(..k) above establish k entries; t is below k"
            )]
            let rjt = r_j[t];
            diagonal += rjt * rjt;
            *slot = -rjt * rtj;
        }
        *row.get_mut(t)? = diagonal;
    }
    for slot in p.iter_mut() {
        *slot = 1.0 / k as f64;
    }
    let mut qp = vec![0.0; k];
    for _iter in 0..max_iter {
        let mut p_qp = 0.0;
        for ((qp_t, &p_t), row) in qp.iter_mut().zip(p.iter()).zip(&q_mat) {
            *qp_t = 0.0;
            for (&q, &p_j) in row.iter().zip(p.iter()) {
                *qp_t += q * p_j;
            }
            p_qp += p_t * *qp_t;
        }
        let mut max_error = 0.0;
        for &value in &qp {
            let error = (value - p_qp).abs();
            if error > max_error {
                max_error = error;
            }
        }
        if max_error < eps {
            break;
        }
        for (t, row) in q_mat.iter().enumerate() {
            let qp_t = *qp.get(t)?;
            let diagonal = *row.get(t)?;
            let diff = (-qp_t + p_qp) / diagonal;
            *p.get_mut(t)? += diff;
            p_qp = (p_qp + diff * (diff * diagonal + 2.0 * qp_t)) / (1.0 + diff) / (1.0 + diff);
            for ((qp_j, p_j), &q) in qp.iter_mut().zip(p.iter_mut()).zip(row) {
                *qp_j = (*qp_j + diff * q) / (1.0 + diff);
                *p_j /= 1.0 + diff;
            }
        }
    }
    Some(())
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
) -> Option<Vec<f64>> {
    let l = prob.labels.len();

    // Build training sub-problem (exclude held-out fold)
    if prob.instances.len() != l || perm.len() != l {
        return None;
    }
    let held_len = end.checked_sub(begin)?;
    let sub_len = l.checked_sub(held_len)?;
    let held = perm.get(begin..end)?;
    let mut sub_instances = Vec::with_capacity(sub_len);
    let mut sub_labels = Vec::with_capacity(sub_len);

    for &pi in perm.get(..begin)? {
        sub_instances.push(prob.instances.get(pi)?.clone());
        sub_labels.push(*prob.labels.get(pi)?);
    }
    for &pi in perm.get(end..l)? {
        sub_instances.push(prob.instances.get(pi)?.clone());
        sub_labels.push(*prob.labels.get(pi)?);
    }

    // Count classes in training set
    let p_count = sub_labels.iter().filter(|&&y| y > 0.0).count();
    let n_count = sub_labels.len().checked_sub(p_count)?;

    if p_count == 0 && n_count == 0 {
        Some(vec![0.0; held_len])
    } else if p_count > 0 && n_count == 0 {
        Some(vec![1.0; held_len])
    } else if p_count == 0 && n_count > 0 {
        Some(vec![-1.0; held_len])
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

        let sign = f64::from(*submodel.label.first()?);
        held.iter()
            .map(|&j| {
                let mut dv = [0.0];
                predict_values(&submodel, prob.instances.get(j)?, &mut dv);
                // Sign correction: ensure +1/−1 ordering
                Some(dv[0] * sign)
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
/// # Panics
///
/// Panics if row lengths differ, rows exceed 2^53, fold arithmetic overflows, or fold training rejects its inputs.
///
/// # Example
///
/// ```
/// use libsvm_rs::io::load_problem_from_reader;
/// use libsvm_rs::{KernelType, SvmParameter, SvmType};
/// let problem = load_problem_from_reader(&b"-1 1:-2\n-1 1:-1\n1 1:1\n1 1:2\n"[..])?;
/// let param = SvmParameter { kernel_type: KernelType::Linear, ..Default::default() };
/// let (a, b) = libsvm_rs::probability::svm_binary_svc_probability(&problem, &param, 1.0, 1.0);
/// assert!(a.is_finite() && b.is_finite());
/// Ok::<(), libsvm_rs::SvmError>(())
/// ```
#[must_use]
pub fn svm_binary_svc_probability(
    prob: &SvmProblem,
    param: &SvmParameter,
    cp: f64,
    cn: f64,
) -> (f64, f64) {
    let parameters = try_binary_svc_probability(prob, param, cp, cn);
    #[expect(
        clippy::expect_used,
        reason = "try_binary_svc_probability checks row lengths, fold arithmetic and permutation slices"
    )]
    parameters.expect(
        "binary probability requires consistent row lengths, fold offsets and at most 2^53 rows; pass one label per instance",
    )
}

fn try_binary_svc_probability(
    prob: &SvmProblem,
    param: &SvmParameter,
    cp: f64,
    cn: f64,
) -> Option<(f64, f64)> {
    let l = prob.labels.len();
    if prob.instances.len() != l || l as u128 > (1u128 << 53) {
        return None;
    }
    let nr_fold = 5.min(l.max(1));
    let mut perm: Vec<usize> = (0..l).collect();
    let mut dec_values = vec![0.0; l];
    for i in 0..l {
        let remaining = l.checked_sub(i)?;
        let j = i.checked_add(c_rand().checked_rem(remaining)?)?;
        perm.swap(i, j);
    }
    let bounds: Vec<(usize, usize)> = (0..nr_fold)
        .map(|fold: usize| {
            Some((
                fold.checked_mul(l)?.checked_div(nr_fold)?,
                fold.checked_add(1)?.checked_mul(l)?.checked_div(nr_fold)?,
            ))
        })
        .collect::<Option<_>>()?;
    #[cfg(feature = "rayon")]
    let predictions: Vec<Vec<f64>> = bounds
        .par_iter()
        .map(|&(begin, end)| {
            evaluate_binary_svc_probability_fold(prob, param, cp, cn, &perm, begin, end)
        })
        .collect::<Option<_>>()?;
    #[cfg(not(feature = "rayon"))]
    let predictions: Vec<Vec<f64>> = bounds
        .iter()
        .map(|&(begin, end)| {
            evaluate_binary_svc_probability_fold(prob, param, cp, cn, &perm, begin, end)
        })
        .collect::<Option<_>>()?;
    let mut indices = perm.iter();
    for fold in predictions {
        for prediction in fold {
            *dec_values.get_mut(*indices.next()?)? = prediction;
        }
    }
    try_sigmoid_train(&dec_values, &prob.labels)
}

// ─── One-class probability ───────────────────────────────────────────

/// Predict probability for one-class SVM from density marks.
///
/// Bin-lookup in precomputed density marks (10 entries). Returns a
/// probability estimate in (0, 1).
/// Counts from manually supplied slices above 2^53 round when converted to `f64`.
///
/// Matches LIBSVM's `predict_one_class_probability`.
///
/// # Example
///
/// ```
/// use libsvm_rs::probability::predict_one_class_probability;
/// let marks = [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9];
/// let probability = predict_one_class_probability(&marks, 0.0);
/// assert_eq!(probability.to_bits(), 0.5_f64.to_bits());
/// Ok::<(), libsvm_rs::SvmError>(())
/// ```
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    reason = "C converts the same integer counts to double at these points"
)]
pub fn predict_one_class_probability(prob_density_marks: &[f64], dec_value: f64) -> f64 {
    let nr_marks = prob_density_marks.len();
    if nr_marks == 0 {
        return 0.5;
    }

    if prob_density_marks
        .first()
        .is_some_and(|&first| dec_value < first)
    {
        return 0.001;
    }
    if prob_density_marks
        .last()
        .is_some_and(|&last| dec_value > last)
    {
        return 0.999;
    }

    for (i, &mark) in prob_density_marks.iter().enumerate().skip(1) {
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
/// # Panics
///
/// Panics if row lengths differ, mark arithmetic overflows, or prediction rejects the model.
///
/// # Example
///
/// ```
/// use libsvm_rs::io::load_problem_from_reader;
/// use libsvm_rs::{KernelType, SvmParameter, SvmType};
/// let problem = load_problem_from_reader(&b"-1 1:-2\n-1 1:-1\n1 1:1\n1 1:2\n"[..])?;
/// let param = SvmParameter { svm_type: SvmType::OneClass, gamma: 0.5, ..Default::default() };
/// let model = libsvm_rs::train::svm_train(&problem, &param);
/// let marks = libsvm_rs::probability::svm_one_class_probability(&problem, &model);
/// assert!(marks.is_none()); // Four rows cannot supply five positive and five negative decisions.
/// Ok::<(), libsvm_rs::SvmError>(())
/// ```
#[must_use]
pub fn svm_one_class_probability(prob: &SvmProblem, model: &SvmModel) -> Option<Vec<f64>> {
    let marks = try_one_class_probability(prob, model);
    #[expect(
        clippy::expect_used,
        reason = "try_one_class_probability checks row lengths and density mark indices"
    )]
    marks.expect(
        "one-class probability requires consistent row lengths and representable mark indices; pass one label per instance",
    )
}

fn try_one_class_probability(
    prob: &SvmProblem,
    model: &SvmModel,
) -> Result<Option<Vec<f64>>, &'static str> {
    (|| {
        let l = prob.labels.len();
        if prob.instances.len() != l {
            return None;
        }
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
        let pos_counter = l.checked_sub(neg_counter)?;

        let nr_marks: usize = 10;
        let mid = nr_marks / 2; // 5

        if neg_counter < mid || pos_counter < mid {
            crate::info(&format!(
                "WARNING: number of positive or negative decision values <{mid}; \
             too few to do a probability estimation.\n"
            ));
            return Some(None);
        }

        let mut tmp_marks = vec![0.0; nr_marks.checked_add(1)?];

        for (i, slot) in tmp_marks.get_mut(..mid)?.iter_mut().enumerate() {
            *slot = *dec_values.get(i.checked_mul(neg_counter)?.checked_div(mid)?)?;
        }
        *tmp_marks.get_mut(mid)? = 0.0;
        for (i, slot) in tmp_marks.iter_mut().enumerate().skip(mid.checked_add(1)?) {
            let index = neg_counter.checked_sub(1)?.checked_add(
                i.checked_sub(mid)?
                    .checked_mul(pos_counter)?
                    .checked_div(mid)?,
            )?;
            *slot = *dec_values.get(index)?;
        }
        let mut marks = vec![0.0; nr_marks];
        for (slot, pair) in marks.iter_mut().zip(tmp_marks.windows(2)) {
            let [left, right] = pair else {
                return None;
            };
            *slot = (left + right) / 2.0;
        }
        Some(Some(marks))
    })()
    .ok_or("inconsistent row lengths or density mark indices; pass one label per instance")
}

// ─── SVR probability ─────────────────────────────────────────────────

/// Estimate Laplace scale parameter for SVR probability.
///
/// Performs 5-fold CV to get residuals, computes MAE, then applies
/// outlier rejection (exclude |residual| > 5·√(2·mae²)) and
/// recomputes. Returns the final MAE (= σ of Laplace distribution).
///
/// Matches LIBSVM's `svm_svr_probability`.
#[must_use]
/// # Panics
///
/// Panics if row lengths differ, rows exceed 2^53, or cross-validation rejects a fold.
///
/// # Example
///
/// ```
/// use libsvm_rs::io::load_problem_from_reader;
/// use libsvm_rs::{KernelType, SvmParameter, SvmType};
/// let problem = load_problem_from_reader(&b"-1 1:-2\n-1 1:-1\n1 1:1\n1 1:2\n"[..])?;
/// let param = SvmParameter {
///     svm_type: SvmType::EpsilonSvr,
///     kernel_type: KernelType::Linear,
///     ..Default::default()
/// };
/// let sigma = libsvm_rs::probability::svm_svr_probability(&problem, &param);
/// assert!(sigma.is_finite() && sigma >= 0.0);
/// Ok::<(), libsvm_rs::SvmError>(())
/// ```
pub fn svm_svr_probability(prob: &SvmProblem, param: &SvmParameter) -> f64 {
    let scale = try_svr_probability(prob, param);
    #[expect(
        clippy::expect_used,
        reason = "try_svr_probability checks row counts; svm_cross_validation returns one prediction per row"
    )]
    scale.expect("SVR probability requires consistent row lengths and at most 2^53 rows; pass one label per instance")
}

#[expect(
    clippy::cast_precision_loss,
    reason = "Row counts are checked against 2^53 before conversion"
)]
fn try_svr_probability(prob: &SvmProblem, param: &SvmParameter) -> Option<f64> {
    let l = prob.labels.len();
    let nr_fold = 5;
    if prob.instances.len() != l || l as u128 > (1u128 << 53) {
        return None;
    }
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
            count = count.checked_add(1)?;
        } else {
            mae += residual.abs();
        }
    }
    mae /= l.checked_sub(count)? as f64;

    crate::info(&format!(
        "Prob. model for test data: target value = predicted value + z,\n\
         z: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma= {mae:.6}\n"
    ));

    Some(mae)
}

// ─── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::load_problem;
    use crate::train::svm_train;
    use crate::types::{KernelType, SvmNode, SvmParameter, SvmProblem, SvmType};
    use std::path::PathBuf;

    #[test]
    fn try_sigmoid_train_rejects_inconsistent_lengths() {
        assert!(try_sigmoid_train(&[0.0], &[]).is_none());
        assert!(try_sigmoid_train(&[], &[1.0]).is_none());
    }

    #[test]
    fn try_multiclass_probability_rejects_inconsistent_dimensions() {
        assert!(try_multiclass_probability(2, &[vec![0.0; 2]], &mut [0.0; 2]).is_none());
        assert!(try_multiclass_probability(2, &[vec![0.0], vec![0.0; 2]], &mut [0.0; 2]).is_none());
        assert!(try_multiclass_probability(2, &[vec![0.0; 2], vec![0.0; 2]], &mut [0.0]).is_none());
    }

    #[test]
    fn evaluate_binary_svc_probability_fold_rejects_inconsistent_rows_and_offsets() {
        let mut problem = SvmProblem {
            labels: vec![1.0],
            instances: Vec::new(),
        };
        let param = SvmParameter::default();
        assert!(
            evaluate_binary_svc_probability_fold(&problem, &param, 1.0, 1.0, &[0], 0, 1).is_none()
        );
        problem.instances.push(Vec::new());
        assert!(
            evaluate_binary_svc_probability_fold(&problem, &param, 1.0, 1.0, &[0], 1, 0).is_none()
        );
        assert!(
            evaluate_binary_svc_probability_fold(&problem, &param, 1.0, 1.0, &[0], 0, 2).is_none()
        );
    }

    #[test]
    fn try_binary_svc_probability_rejects_inconsistent_rows() {
        let problem = SvmProblem {
            labels: vec![1.0],
            instances: Vec::new(),
        };
        assert!(try_binary_svc_probability(&problem, &SvmParameter::default(), 1.0, 1.0).is_none());
    }

    #[test]
    fn try_one_class_probability_rejects_inconsistent_rows() {
        let problem = SvmProblem {
            labels: vec![1.0],
            instances: Vec::new(),
        };
        let model = crate::io::load_model(&data_dir().join("heart_scale.model")).unwrap();
        assert!(try_one_class_probability(&problem, &model).is_err());
    }

    #[test]
    fn try_svr_probability_rejects_inconsistent_rows() {
        let problem = SvmProblem {
            labels: vec![1.0],
            instances: Vec::new(),
        };
        assert!(try_svr_probability(&problem, &SvmParameter::default()).is_none());
    }

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
            "probabilities sum to {sum}, expected ~1.0"
        );
        for &pi in &p {
            assert!(pi > 0.0, "probability should be positive, got {pi}");
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
            let [left, right] = pair else { unreachable!() };
            assert!(left <= right);
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
