//! SVM training pipeline.
//!
//! Provides `svm_train` which produces an `SvmModel` from an `SvmProblem`
//! and `SvmParameter`. Matches the original LIBSVM's `svm_train` function.

#![cfg_attr(
    test,
    expect(
        clippy::unwrap_used,
        reason = "Tests unwrap fixture loading and checked assertions"
    )
)]

use crate::qmatrix::{OneClassQ, SvcQ, SvrQ};
use crate::solver::{SolutionInfo, Solver, SolverVariant};
use crate::types::{SvmModel, SvmNode, SvmParameter, SvmProblem, SvmType};
use crate::util::group_classes;

/// Internal decision function result from one binary sub-problem.
struct DecisionFunction {
    alpha: Vec<f64>,
    rho: f64,
}

fn sign_labels(labels: &[f64]) -> Vec<i8> {
    labels
        .iter()
        .map(|&v| if v > 0.0 { 1 } else { -1 })
        .collect()
}

// Reject counts before converting them to f64 or allocating doubled SVR arrays.
fn solver_rows(rows: usize, multiplier: usize) -> Option<usize> {
    let count = rows.checked_mul(multiplier)?;
    if count as u128 > (1u128 << 53) {
        return None;
    }
    Some(count)
}

// ─── Solve dispatchers ──────────────────────────────────────────────

#[expect(
    clippy::many_single_char_names,
    reason = "The C solver uses these short names"
)]
fn solve_c_svc(
    x: &[Vec<SvmNode>],
    labels: &[f64],
    param: &SvmParameter,
    cp: f64,
    cn: f64,
) -> Option<(Vec<f64>, SolutionInfo)> {
    let l = solver_rows(x.len(), 1)?;
    let mut alpha = vec![0.0; l];
    let p: Vec<f64> = vec![-1.0; l];
    if labels.len() != l {
        return None;
    }
    let y = sign_labels(labels);

    let q = Box::new(SvcQ::new(x, param, &y));
    let si = Solver::solve(
        SolverVariant::Standard,
        l,
        q,
        &p,
        &y,
        &mut alpha,
        cp,
        cn,
        param.eps,
        param.shrinking,
    );

    // Multiply alpha by y to get signed coefficients
    for (a, &sign) in alpha.iter_mut().zip(&y) {
        *a *= f64::from(sign);
    }

    Some((alpha, si))
}

#[expect(
    clippy::cast_precision_loss,
    reason = "C converts the same integer counts to double at these points"
)]
#[expect(
    clippy::many_single_char_names,
    reason = "The C solver uses these short names"
)]
fn solve_nu_svc(
    x: &[Vec<SvmNode>],
    labels: &[f64],
    param: &SvmParameter,
) -> Option<(Vec<f64>, SolutionInfo)> {
    let l = solver_rows(x.len(), 1)?;
    let nu = param.nu;
    if labels.len() != l {
        return None;
    }
    let y = sign_labels(labels);

    // Initialize alpha: spread nu*l/2 among positive and negative samples
    let mut alpha = vec![0.0; l];
    let mut sum_pos = nu * l as f64 / 2.0;
    let mut sum_neg = nu * l as f64 / 2.0;
    for (a, &sign) in alpha.iter_mut().zip(&y) {
        if sign == 1 {
            *a = f64::min(1.0, sum_pos);
            sum_pos -= *a;
        } else {
            *a = f64::min(1.0, sum_neg);
            sum_neg -= *a;
        }
    }

    let p = vec![0.0; l];
    let q = Box::new(SvcQ::new(x, param, &y));
    let mut si = Solver::solve(
        SolverVariant::Nu,
        l,
        q,
        &p,
        &y,
        &mut alpha,
        1.0,
        1.0,
        param.eps,
        param.shrinking,
    );

    let r = si.r;
    for (a, &sign) in alpha.iter_mut().zip(&y) {
        *a *= f64::from(sign) / r;
    }
    si.rho /= r;
    si.obj /= r * r;
    si.upper_bound_p = 1.0 / r;
    si.upper_bound_n = 1.0 / r;

    Some((alpha, si))
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "C truncates the same values at these points"
)]
#[expect(
    clippy::cast_precision_loss,
    reason = "C converts the same integer counts to double at these points"
)]
#[expect(
    clippy::cast_sign_loss,
    reason = "C converts the same signed values at these points"
)]
#[expect(
    clippy::many_single_char_names,
    reason = "The C solver uses these short names"
)]
fn solve_one_class(x: &[Vec<SvmNode>], param: &SvmParameter) -> Option<(Vec<f64>, SolutionInfo)> {
    let l = solver_rows(x.len(), 1)?;

    // Initialize alpha: first n=floor(nu*l) at 1, fractional remainder, rest 0
    let n = (param.nu * l as f64) as usize;
    let mut alpha = vec![0.0; l];
    for a in alpha.iter_mut().take(n.min(l)) {
        *a = 1.0;
    }
    if n < l {
        *alpha.get_mut(n)? = param.nu * l as f64 - n as f64;
    }

    let p = vec![0.0; l];
    let y = vec![1i8; l];
    let q = Box::new(OneClassQ::new(x, param));
    let si = Solver::solve(
        SolverVariant::Standard,
        l,
        q,
        &p,
        &y,
        &mut alpha,
        1.0,
        1.0,
        param.eps,
        param.shrinking,
    );

    Some((alpha, si))
}

fn solve_epsilon_svr(
    x: &[Vec<SvmNode>],
    labels: &[f64],
    param: &SvmParameter,
) -> Option<(Vec<f64>, SolutionInfo)> {
    let l = solver_rows(x.len(), 1)?;
    if labels.len() != l {
        return None;
    }
    let doubled = solver_rows(l, 2)?;
    let mut alpha2 = vec![0.0; doubled];
    let mut linear_term = vec![0.0; doubled];
    let mut y = vec![0i8; doubled];

    let (linear_pos, linear_neg) = linear_term.split_at_mut_checked(l)?;
    let (y_pos, y_neg) = y.split_at_mut_checked(l)?;
    for (((lp, ln), (yp, yn)), &label) in linear_pos
        .iter_mut()
        .zip(linear_neg)
        .zip(y_pos.iter_mut().zip(y_neg))
        .zip(labels)
    {
        *lp = param.p - label;
        *yp = 1;
        *ln = param.p + label;
        *yn = -1;
    }

    let q = Box::new(SvrQ::new(x, param));
    let si = Solver::solve(
        SolverVariant::Standard,
        doubled,
        q,
        &linear_term,
        &y,
        &mut alpha2,
        param.c,
        param.c,
        param.eps,
        param.shrinking,
    );

    let mut alpha = vec![0.0; l];
    let (positive, negative) = alpha2.split_at_checked(l)?;
    for (a, (&pos, &neg)) in alpha.iter_mut().zip(positive.iter().zip(negative)) {
        *a = pos - neg;
    }

    Some((alpha, si))
}

#[expect(
    clippy::cast_precision_loss,
    reason = "C converts the same integer counts to double at these points"
)]
#[expect(
    clippy::many_single_char_names,
    reason = "The C solver uses these short names"
)]
fn solve_nu_svr(
    x: &[Vec<SvmNode>],
    labels: &[f64],
    param: &SvmParameter,
) -> Option<(Vec<f64>, SolutionInfo)> {
    let l = solver_rows(x.len(), 1)?;
    let c = param.c;
    if labels.len() != l {
        return None;
    }
    let doubled = solver_rows(l, 2)?;
    let mut alpha2 = vec![0.0; doubled];
    let mut linear_term = vec![0.0; doubled];
    let mut y = vec![0i8; doubled];

    let mut sum = c * param.nu * l as f64 / 2.0;
    let (alpha_pos, alpha_neg) = alpha2.split_at_mut_checked(l)?;
    let (linear_pos, linear_neg) = linear_term.split_at_mut_checked(l)?;
    let (y_pos, y_neg) = y.split_at_mut_checked(l)?;
    for ((((ap, an), (lp, ln)), (yp, yn)), &label) in alpha_pos
        .iter_mut()
        .zip(alpha_neg)
        .zip(linear_pos.iter_mut().zip(linear_neg))
        .zip(y_pos.iter_mut().zip(y_neg))
        .zip(labels)
    {
        let a = f64::min(sum, c);
        *ap = a;
        *an = a;
        sum -= a;
        *lp = -label;
        *yp = 1;
        *ln = label;
        *yn = -1;
    }

    let q = Box::new(SvrQ::new(x, param));
    let si = Solver::solve(
        SolverVariant::Nu,
        doubled,
        q,
        &linear_term,
        &y,
        &mut alpha2,
        c,
        c,
        param.eps,
        param.shrinking,
    );

    let mut alpha = vec![0.0; l];
    let (positive, negative) = alpha2.split_at_checked(l)?;
    for (a, (&pos, &neg)) in alpha.iter_mut().zip(positive.iter().zip(negative)) {
        *a = pos - neg;
    }

    Some((alpha, si))
}

// ─── svm_train_one ──────────────────────────────────────────────────

#[expect(
    clippy::similar_names,
    reason = "Keep the distinct C names for comparison with LIBSVM"
)]
fn svm_train_one(
    x: &[Vec<SvmNode>],
    labels: &[f64],
    param: &SvmParameter,
    cp: f64,
    cn: f64,
) -> Option<DecisionFunction> {
    if x.len() != labels.len() {
        return None;
    }
    let (alpha, si) = match param.svm_type {
        SvmType::CSvc => solve_c_svc(x, labels, param, cp, cn),
        SvmType::NuSvc => solve_nu_svc(x, labels, param),
        SvmType::OneClass => solve_one_class(x, param),
        SvmType::EpsilonSvr => solve_epsilon_svr(x, labels, param),
        SvmType::NuSvr => solve_nu_svr(x, labels, param),
    }?;

    crate::info(&format!("obj = {:.6}, rho = {:.6}\n", si.obj, si.rho));

    // Count SVs
    let n_sv = alpha.iter().filter(|a| a.abs() > 0.0).count();
    let n_bsv = alpha
        .iter()
        .zip(labels)
        .filter(|&(a, &label)| {
            if a.abs() > 0.0 {
                if label > 0.0 {
                    a.abs() >= si.upper_bound_p
                } else {
                    a.abs() >= si.upper_bound_n
                }
            } else {
                false
            }
        })
        .count();
    crate::info(&format!("nSV = {n_sv}, nBSV = {n_bsv}\n"));

    Some(DecisionFunction { alpha, rho: si.rho })
}

fn mark_nonzero_indices(nonzero: &mut [bool], start: usize, alphas: &[f64]) -> Option<()> {
    let end = start.checked_add(alphas.len())?;
    let slots = nonzero.get_mut(start..end)?;
    for (slot, &alpha) in slots.iter_mut().zip(alphas) {
        if !*slot && alpha.abs() > 0.0 {
            *slot = true;
        }
    }
    Some(())
}

fn count_nonzero(nonzero: &[bool], start: usize, len: usize) -> Option<usize> {
    let end = start.checked_add(len)?;
    Some(nonzero.get(start..end)?.iter().filter(|&&v| v).count())
}

// ─── svm_train ──────────────────────────────────────────────────────

/// Train an SVM model from a problem and parameters.
///
/// Matches LIBSVM's `svm_train` function. Produces an `SvmModel` that
/// can be used for prediction or saved to a file.
///
/// ```
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
///
/// let model = svm_train(&problem, &param);
/// assert_eq!(model.svm_type(), SvmType::CSvc);
/// # Ok::<(), libsvm_rs::SvmError>(())
/// ```
/// # Panics
///
/// Panics for empty classification input, unequal label and instance lengths, overflowing counts,
/// solver rows above 2^53, or inconsistent internal class offsets.
#[must_use]
pub fn svm_train(problem: &SvmProblem, param: &SvmParameter) -> SvmModel {
    // Compute effective gamma if zero
    let mut param = param.clone();
    if param.gamma == 0.0 && !problem.instances.is_empty() {
        let max_index = problem
            .instances
            .iter()
            .flat_map(|inst| inst.iter())
            .map(|n| n.index)
            .max()
            .unwrap_or(0);
        if max_index > 0 {
            param.gamma = 1.0 / f64::from(max_index);
        }
    }

    let model = match param.svm_type {
        SvmType::OneClass | SvmType::EpsilonSvr | SvmType::NuSvr => {
            train_regression_or_one_class(problem, &param)
        }
        SvmType::CSvc | SvmType::NuSvc => train_classification(problem, &param),
    };
    // Both paths check row lengths, counts and class slices before using them.
    #[expect(clippy::expect_used, reason = "private training paths check row lengths, counts and class slices")]
    model.expect("training requires consistent row lengths and offsets, representable counts, and at most 2^53 solver rows. Pass one label per instance and at least one row for classification")
}

fn train_regression_or_one_class(problem: &SvmProblem, param: &SvmParameter) -> Option<SvmModel> {
    let f = svm_train_one(&problem.instances, &problem.labels, param, 0.0, 0.0)?;

    // Extract support vectors
    let mut sv = Vec::new();
    let mut sv_coef = Vec::new();
    let mut sv_indices = Vec::new();

    if f.alpha.len() != problem.instances.len() {
        return None;
    }
    for (i, (instance, &a)) in problem.instances.iter().zip(&f.alpha).enumerate() {
        if a.abs() > 0.0 {
            sv.push(instance.clone());
            sv_coef.push(a);
            sv_indices.push(i.checked_add(1)?);
        }
    }

    let mut model = SvmModel {
        param: param.clone(),
        nr_class: 2,
        sv,
        sv_coef: vec![sv_coef],
        rho: vec![f.rho],
        prob_a: Vec::new(),
        prob_b: Vec::new(),
        prob_density_marks: Vec::new(),
        sv_indices,
        label: Vec::new(),
        n_sv: Vec::new(),
    };

    // Probability estimates
    if param.probability {
        match param.svm_type {
            SvmType::EpsilonSvr | SvmType::NuSvr => {
                model.prob_a = vec![crate::probability::svm_svr_probability(problem, param)];
            }
            SvmType::OneClass => {
                if let Some(marks) = crate::probability::svm_one_class_probability(problem, &model)
                {
                    model.prob_density_marks = marks;
                }
            }
            _ => {}
        }
    }

    Some(model)
}

#[expect(clippy::too_many_lines, reason = "Keep the source loop in one place")]
fn train_classification(problem: &SvmProblem, param: &SvmParameter) -> Option<SvmModel> {
    let l = problem.instances.len();
    if problem.labels.len() != l || l == 0 || l as u128 > (1u128 << 53) {
        return None;
    }
    let group = group_classes(&problem.labels)?;
    let nr_class = group.label.len();

    if nr_class == 1 {
        crate::info("WARNING: training data in only one class. See README for details.\n");
    }

    // Reorder instances by class
    let x: Vec<&Vec<SvmNode>> = group
        .perm
        .iter()
        .map(|&i| problem.instances.get(i))
        .collect::<Option<_>>()?;

    // Calculate weighted C
    let mut weighted_c = vec![param.c; nr_class];
    for &(wlabel, wval) in &param.weight {
        if let Some(j) = group.label.iter().position(|&lab| lab == wlabel) {
            *weighted_c.get_mut(j)? *= wval;
        } else {
            crate::info(&format!(
                "WARNING: class label {wlabel} specified in weight is not found\n"
            ));
        }
    }

    // Train k*(k-1)/2 binary classifiers
    let mut nonzero = vec![false; l];
    let coef_rows = nr_class.checked_sub(1)?;
    let n_pairs = nr_class.checked_mul(coef_rows)? / 2;
    let mut decisions = Vec::with_capacity(n_pairs);

    // Probability arrays (filled only when param.probability is set)
    let mut prob_a = Vec::new();
    let mut prob_b = Vec::new();
    if param.probability {
        prob_a.reserve(n_pairs);
        prob_b.reserve(n_pairs);
    }

    for i in 0..nr_class {
        for j in i.checked_add(1)?..nr_class {
            let si = *group.start.get(i)?;
            let sj = *group.start.get(j)?;
            let ci = *group.count.get(i)?;
            let cj = *group.count.get(j)?;

            // Build sub-problem
            let sub_len = ci.checked_add(cj)?;
            let mut sub_x = Vec::with_capacity(sub_len);
            let mut sub_labels = Vec::with_capacity(sub_len);
            for instance in x.get(si..si.checked_add(ci)?)? {
                sub_x.push((*instance).clone());
                sub_labels.push(1.0);
            }
            for instance in x.get(sj..sj.checked_add(cj)?)? {
                sub_x.push((*instance).clone());
                sub_labels.push(-1.0);
            }

            // Probability estimates via internal 5-fold CV (before final training)
            if param.probability {
                let sub_prob = SvmProblem {
                    labels: sub_labels.clone(),
                    instances: sub_x.clone(),
                };
                let (pa, pb) = crate::probability::svm_binary_svc_probability(
                    &sub_prob,
                    param,
                    *weighted_c.get(i)?,
                    *weighted_c.get(j)?,
                );
                prob_a.push(pa);
                prob_b.push(pb);
            }

            let f = svm_train_one(
                &sub_x,
                &sub_labels,
                param,
                *weighted_c.get(i)?,
                *weighted_c.get(j)?,
            )?;

            // Mark nonzero alphas
            let (alpha_i, alpha_j) = f.alpha.split_at_checked(ci)?;
            if alpha_j.len() != cj {
                return None;
            }
            mark_nonzero_indices(&mut nonzero, si, alpha_i)?;
            mark_nonzero_indices(&mut nonzero, sj, alpha_j)?;

            decisions.push(f);
        }
    }

    // Build model output
    let labels: Vec<i32> = group.label.clone();
    let rho: Vec<f64> = decisions.iter().map(|d| d.rho).collect();

    // Count SVs per class
    let mut total_sv = 0usize;
    let mut n_sv_per_class = vec![0usize; nr_class];
    for (i, n_sv) in n_sv_per_class.iter_mut().enumerate().take(nr_class) {
        let n = count_nonzero(&nonzero, *group.start.get(i)?, *group.count.get(i)?)?;
        total_sv = total_sv.checked_add(n)?;
        *n_sv = n;
    }

    crate::info(&format!("Total nSV = {total_sv}\n"));

    // Collect SVs and indices
    let mut model_sv = Vec::with_capacity(total_sv);
    let mut model_sv_indices = Vec::with_capacity(total_sv);
    for ((&is_nonzero, instance), &original) in nonzero.iter().zip(&x).zip(&group.perm) {
        if is_nonzero {
            model_sv.push((*instance).clone());
            model_sv_indices.push(original.checked_add(1)?);
        }
    }

    // Build nz_start (cumulative start of nonzero SVs per class)
    let mut nz_start = vec![0usize; nr_class];
    let mut cumulative = 0usize;
    for (slot, &n) in nz_start.iter_mut().zip(&n_sv_per_class) {
        *slot = cumulative;
        cumulative = cumulative.checked_add(n)?;
    }

    // Build sv_coef matrix: (nr_class - 1) rows × total_sv columns
    let mut sv_coef = vec![vec![0.0; total_sv]; coef_rows];

    let mut decision_iter = decisions.iter();
    for i in 0..nr_class {
        for j in i.checked_add(1)?..nr_class {
            let si = *group.start.get(i)?;
            let sj = *group.start.get(j)?;
            let ci = *group.count.get(i)?;
            let cj = *group.count.get(j)?;
            let decision = decision_iter.next()?;
            let (alpha_i, alpha_j) = decision.alpha.split_at_checked(ci)?;
            if alpha_j.len() != cj {
                return None;
            }
            for (class, start, count, alphas, row_index) in [
                (i, si, ci, alpha_i, j.checked_sub(1)?),
                (j, sj, cj, alpha_j, i),
            ] {
                let flags = nonzero.get(start..start.checked_add(count)?)?;
                let offset = *nz_start.get(class)?;
                let n = *n_sv_per_class.get(class)?;
                let mut output = sv_coef
                    .get_mut(row_index)?
                    .get_mut(offset..offset.checked_add(n)?)?
                    .iter_mut();
                for (&flag, &alpha) in flags.iter().zip(alphas) {
                    if flag {
                        *output.next()? = alpha;
                    }
                }
            }
        }
    }

    Some(SvmModel {
        param: param.clone(),
        nr_class,
        sv: model_sv,
        sv_coef,
        rho,
        prob_a,
        prob_b,
        prob_density_marks: Vec::new(),
        sv_indices: model_sv_indices,
        label: labels,
        n_sv: n_sv_per_class,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{load_model, load_problem};
    use crate::predict::predict;
    use crate::types::KernelType;
    use std::path::PathBuf;

    #[test]
    fn solve_c_svc_rejects_mismatched_rows() {
        assert!(solve_c_svc(&[], &[1.0], &SvmParameter::default(), 1.0, 1.0).is_none());
    }

    #[test]
    fn solve_nu_svc_rejects_mismatched_rows() {
        assert!(solve_nu_svc(&[], &[1.0], &SvmParameter::default()).is_none());
    }

    #[test]
    fn solve_epsilon_svr_rejects_mismatched_rows() {
        assert!(solve_epsilon_svr(&[], &[1.0], &SvmParameter::default()).is_none());
    }

    #[test]
    fn solve_nu_svr_rejects_mismatched_rows() {
        assert!(solve_nu_svr(&[], &[1.0], &SvmParameter::default()).is_none());
    }

    #[test]
    fn svm_train_one_rejects_mismatched_rows() {
        assert!(svm_train_one(&[], &[1.0], &SvmParameter::default(), 1.0, 1.0).is_none());
    }

    #[test]
    fn train_regression_or_one_class_rejects_mismatched_rows() {
        let problem = SvmProblem {
            labels: vec![1.0],
            instances: Vec::new(),
        };
        assert!(train_regression_or_one_class(&problem, &SvmParameter::default()).is_none());
    }

    #[test]
    fn train_classification_rejects_mismatched_rows() {
        // Two instances and one label: the empty-input check does not apply,
        // so only the length mismatch can reject.
        let problem = SvmProblem {
            labels: vec![1.0],
            instances: vec![Vec::new(), Vec::new()],
        };
        assert!(train_classification(&problem, &SvmParameter::default()).is_none());
    }

    #[test]
    fn solver_rows_rejects_overflow() {
        assert!(solver_rows(usize::MAX, 2).is_none());
        if let Ok(limit) = usize::try_from(1u128 << 53) {
            assert_eq!(solver_rows(limit, 1), Some(limit));
            assert!(solver_rows(limit.checked_add(1).unwrap(), 1).is_none());
            assert!(solver_rows(limit, 2).is_none());
        }
    }

    #[test]
    fn mark_nonzero_indices_rejects_invalid_window() {
        assert!(mark_nonzero_indices(&mut [false], 1, &[1.0]).is_none());
        assert!(mark_nonzero_indices(&mut [false], usize::MAX, &[1.0]).is_none());
    }

    #[test]
    fn count_nonzero_rejects_invalid_window() {
        assert!(count_nonzero(&[false], 1, 1).is_none());
        assert!(count_nonzero(&[false], usize::MAX, 1).is_none());
    }

    fn data_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("data")
    }

    #[test]
    #[expect(
        clippy::cast_possible_wrap,
        reason = "C converts the same values at these points"
    )]
    #[expect(
        clippy::cast_precision_loss,
        reason = "C converts the same integer counts to double at these points"
    )]
    #[expect(
        clippy::float_cmp,
        reason = "Predicted labels are compared with class labels and with the other model"
    )]
    fn train_c_svc_heart_scale() {
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

        let model = svm_train(&problem, &param);

        // Check basic model structure
        assert_eq!(model.nr_class, 2);
        assert_eq!(model.label, vec![1, -1]);
        assert_ne!(model.sv.len(), 0, "model has no support vectors");

        // Compare with C reference model
        let ref_model = load_model(&data_dir().join("heart_scale_ref.model")).unwrap();

        // Same number of SVs (within tolerance — solver iterations may vary slightly)
        let sv_diff = (model.sv.len() as i64 - ref_model.sv.len() as i64).unsigned_abs();
        assert!(
            sv_diff <= 2,
            "SV count mismatch: Rust={}, C={}",
            model.sv.len(),
            ref_model.sv.len()
        );

        // Same rho (within tolerance)
        // Tolerance 1e-4 is a loose bound. The measured difference to the C model is 1 ULP (5.6e-17) here.
        assert!(
            (*model.rho.first().unwrap() - *ref_model.rho.first().unwrap()).abs() < 1e-4,
            "rho mismatch: Rust={}, C={}",
            *model.rho.first().unwrap(),
            *ref_model.rho.first().unwrap()
        );

        // Predictions should match on training data
        let mut correct = 0;
        for (i, instance) in problem.instances.iter().enumerate() {
            let pred = predict(&model, instance);
            if pred == *problem.labels.get(i).unwrap() {
                correct = i32::checked_add(correct, 1).unwrap();
            }
        }
        let accuracy = f64::from(correct) / problem.labels.len() as f64;
        assert!(
            accuracy > 0.85,
            "training accuracy {:.2}% too low",
            accuracy * 100.0
        );

        // Predictions from Rust-trained model should match C-trained model
        let mut mismatches = 0;
        for instance in &problem.instances {
            let rust_pred = predict(&model, instance);
            let c_pred = predict(&ref_model, instance);
            if rust_pred != c_pred {
                mismatches = i32::checked_add(mismatches, 1).unwrap();
            }
        }
        assert!(
            mismatches <= 3,
            "{mismatches} prediction mismatches between Rust-trained and C-trained models"
        );
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
    fn train_c_svc_iris_multiclass() {
        let problem = load_problem(&data_dir().join("iris.scale")).unwrap();
        let param = SvmParameter {
            svm_type: SvmType::CSvc,
            kernel_type: KernelType::Rbf,
            gamma: 0.25, // 1/num_features = 1/4
            c: 1.0,
            cache_size: 100.0,
            eps: 0.001,
            shrinking: true,
            ..Default::default()
        };

        let model = svm_train(&problem, &param);

        // Iris has 3 classes
        assert_eq!(model.nr_class, 3);
        assert_eq!(model.label.len(), 3);
        // 3 class pairs = 3 rho values
        assert_eq!(model.rho.len(), 3);
        // sv_coef has nr_class-1 = 2 rows
        assert_eq!(model.sv_coef.len(), 2);
        // n_sv has 3 entries
        assert_eq!(model.n_sv.len(), 3);

        // Predict on training set — should be very accurate for iris
        let mut correct = 0;
        for (i, instance) in problem.instances.iter().enumerate() {
            let pred = predict(&model, instance);
            if pred == *problem.labels.get(i).unwrap() {
                correct = i32::checked_add(correct, 1).unwrap();
            }
        }
        let accuracy = f64::from(correct) / problem.labels.len() as f64;
        assert!(
            accuracy > 0.95,
            "iris accuracy {:.2}% too low (expected >95%)",
            accuracy * 100.0
        );
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
    fn train_c_svc_precomputed_kernel() {
        let problem = load_problem(&data_dir().join("heart_scale.precomputed")).unwrap();
        let param = SvmParameter {
            svm_type: SvmType::CSvc,
            kernel_type: KernelType::Precomputed,
            c: 1.0,
            cache_size: 100.0,
            eps: 0.001,
            shrinking: true,
            ..Default::default()
        };

        let model = svm_train(&problem, &param);

        assert_eq!(model.nr_class, 2);
        assert_ne!(model.sv.len(), 0, "model has no support vectors");

        // Sanity-check predictions on training data.
        let mut correct = 0;
        for (i, instance) in problem.instances.iter().enumerate() {
            let pred = predict(&model, instance);
            if pred == *problem.labels.get(i).unwrap() {
                correct = i32::checked_add(correct, 1).unwrap();
            }
        }
        let accuracy = f64::from(correct) / problem.labels.len() as f64;
        assert!(
            accuracy > 0.70,
            "precomputed-kernel accuracy {:.2}% too low",
            accuracy * 100.0
        );
    }

    #[test]
    #[expect(
        clippy::cast_precision_loss,
        reason = "C converts the same integer counts to double at these points"
    )]
    fn train_one_class() {
        let problem = load_problem(&data_dir().join("heart_scale")).unwrap();
        let param = SvmParameter {
            svm_type: SvmType::OneClass,
            kernel_type: KernelType::Rbf,
            gamma: 1.0 / 13.0,
            nu: 0.5,
            cache_size: 100.0,
            eps: 0.001,
            shrinking: true,
            ..Default::default()
        };

        let model = svm_train(&problem, &param);

        assert_eq!(model.nr_class, 2);
        assert_ne!(model.sv.len(), 0);
        assert_eq!(model.rho.len(), 1);

        // Predict — most training points should be classified as +1 (inlier)
        let mut inliers = 0;
        for instance in &problem.instances {
            let pred = predict(&model, instance);
            if pred > 0.0 {
                inliers = i32::checked_add(inliers, 1).unwrap();
            }
        }
        let inlier_rate = f64::from(inliers) / problem.instances.len() as f64;
        // With nu=0.5, roughly half should be inliers (nu is upper bound on fraction of outliers)
        assert!(
            inlier_rate > 0.3 && inlier_rate < 0.9,
            "unexpected inlier rate: {:.2}%",
            inlier_rate * 100.0
        );
    }

    #[test]
    #[expect(
        clippy::cast_precision_loss,
        reason = "C converts the same integer counts to double at these points"
    )]
    fn train_epsilon_svr() {
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

        let model = svm_train(&problem, &param);

        assert_eq!(model.nr_class, 2); // SVR always has nr_class=2
        assert_ne!(model.sv.len(), 0);

        // Compute MSE on training set — should be reasonable
        let mut mse = 0.0;
        for (i, instance) in problem.instances.iter().enumerate() {
            let pred = predict(&model, instance);
            let err = pred - *problem.labels.get(i).unwrap();
            mse += err * err;
        }
        mse /= problem.instances.len() as f64;

        // MSE should be finite and reasonable
        assert!(mse.is_finite(), "MSE is not finite");
        assert!(mse < 100.0, "MSE too high: {mse}");
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
    fn train_nu_svc() {
        let problem = load_problem(&data_dir().join("heart_scale")).unwrap();
        let param = SvmParameter {
            svm_type: SvmType::NuSvc,
            kernel_type: KernelType::Rbf,
            gamma: 1.0 / 13.0,
            nu: 0.5,
            cache_size: 100.0,
            eps: 0.001,
            shrinking: true,
            ..Default::default()
        };

        let model = svm_train(&problem, &param);

        assert_eq!(model.nr_class, 2);
        assert_ne!(model.sv.len(), 0);

        let mut correct = 0;
        for (i, instance) in problem.instances.iter().enumerate() {
            let pred = predict(&model, instance);
            if pred == *problem.labels.get(i).unwrap() {
                correct = i32::checked_add(correct, 1).unwrap();
            }
        }
        let accuracy = f64::from(correct) / problem.labels.len() as f64;
        assert!(
            accuracy > 0.70,
            "nu-SVC accuracy {:.2}% too low",
            accuracy * 100.0
        );
    }

    #[test]
    fn train_csvc_with_probability() {
        let problem = load_problem(&data_dir().join("heart_scale")).unwrap();
        let param = SvmParameter {
            svm_type: SvmType::CSvc,
            kernel_type: KernelType::Rbf,
            gamma: 1.0 / 13.0,
            c: 1.0,
            cache_size: 100.0,
            eps: 0.001,
            shrinking: true,
            probability: true,
            ..Default::default()
        };

        let model = svm_train(&problem, &param);

        assert_eq!(model.nr_class, 2);
        assert_eq!(model.prob_a.len(), 1, "binary should have 1 probA");
        assert_eq!(model.prob_b.len(), 1, "binary should have 1 probB");
        assert!(model.prob_a.first().unwrap().is_finite());
        assert!(model.prob_b.first().unwrap().is_finite());
    }

    #[test]
    #[expect(
        clippy::cast_precision_loss,
        reason = "C converts the same integer counts to double at these points"
    )]
    fn train_nu_svr() {
        let problem = load_problem(&data_dir().join("housing_scale")).unwrap();
        let param = SvmParameter {
            svm_type: SvmType::NuSvr,
            kernel_type: KernelType::Rbf,
            gamma: 1.0 / 13.0,
            c: 1.0,
            nu: 0.5,
            cache_size: 100.0,
            eps: 0.001,
            shrinking: true,
            ..Default::default()
        };

        let model = svm_train(&problem, &param);

        assert_eq!(model.nr_class, 2);
        assert_ne!(model.sv.len(), 0);

        let mut mse = 0.0;
        for (i, instance) in problem.instances.iter().enumerate() {
            let pred = predict(&model, instance);
            let err = pred - *problem.labels.get(i).unwrap();
            mse += err * err;
        }
        mse /= problem.instances.len() as f64;

        assert!(mse.is_finite(), "MSE is not finite");
        assert!(mse < 200.0, "MSE too high: {mse}");
    }
}
