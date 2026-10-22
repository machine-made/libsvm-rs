//! SVM training pipeline.
//!
//! Provides `svm_train` which produces an `SvmModel` from an `SvmProblem`
//! and `SvmParameter`. Matches the original LIBSVM's `svm_train` function.

use crate::qmatrix::{OneClassQ, SvcQ, SvrQ};
use crate::solver::{SolutionInfo, Solver, SolverVariant};
use crate::types::*;
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

// ─── Solve dispatchers ──────────────────────────────────────────────

fn solve_c_svc(
    x: &[Vec<SvmNode>],
    labels: &[f64],
    param: &SvmParameter,
    cp: f64,
    cn: f64,
) -> (Vec<f64>, SolutionInfo) {
    let l = x.len();
    let mut alpha = vec![0.0; l];
    let p: Vec<f64> = vec![-1.0; l];
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
    for i in 0..l {
        alpha[i] *= y[i] as f64;
    }

    (alpha, si)
}

fn solve_nu_svc(
    x: &[Vec<SvmNode>],
    labels: &[f64],
    param: &SvmParameter,
) -> (Vec<f64>, SolutionInfo) {
    let l = x.len();
    let nu = param.nu;
    let y = sign_labels(labels);

    // Initialize alpha: spread nu*l/2 among positive and negative samples
    let mut alpha = vec![0.0; l];
    let mut sum_pos = nu * l as f64 / 2.0;
    let mut sum_neg = nu * l as f64 / 2.0;
    for i in 0..l {
        if y[i] == 1 {
            alpha[i] = f64::min(1.0, sum_pos);
            sum_pos -= alpha[i];
        } else {
            alpha[i] = f64::min(1.0, sum_neg);
            sum_neg -= alpha[i];
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
    for i in 0..l {
        alpha[i] *= y[i] as f64 / r;
    }
    si.rho /= r;
    si.obj /= r * r;
    si.upper_bound_p = 1.0 / r;
    si.upper_bound_n = 1.0 / r;

    (alpha, si)
}

fn solve_one_class(x: &[Vec<SvmNode>], param: &SvmParameter) -> (Vec<f64>, SolutionInfo) {
    let l = x.len();

    // Initialize alpha: first n=floor(nu*l) at 1, fractional remainder, rest 0
    let n = (param.nu * l as f64) as usize;
    let mut alpha = vec![0.0; l];
    for a in alpha.iter_mut().take(n.min(l)) {
        *a = 1.0;
    }
    if n < l {
        alpha[n] = param.nu * l as f64 - n as f64;
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

    (alpha, si)
}

fn solve_epsilon_svr(
    x: &[Vec<SvmNode>],
    labels: &[f64],
    param: &SvmParameter,
) -> (Vec<f64>, SolutionInfo) {
    let l = x.len();
    let mut alpha2 = vec![0.0; 2 * l];
    let mut linear_term = vec![0.0; 2 * l];
    let mut y = vec![0i8; 2 * l];

    for i in 0..l {
        linear_term[i] = param.p - labels[i];
        y[i] = 1;
        linear_term[i + l] = param.p + labels[i];
        y[i + l] = -1;
    }

    let q = Box::new(SvrQ::new(x, param));
    let si = Solver::solve(
        SolverVariant::Standard,
        2 * l,
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
    for i in 0..l {
        alpha[i] = alpha2[i] - alpha2[i + l];
    }

    (alpha, si)
}

fn solve_nu_svr(
    x: &[Vec<SvmNode>],
    labels: &[f64],
    param: &SvmParameter,
) -> (Vec<f64>, SolutionInfo) {
    let l = x.len();
    let c = param.c;
    let mut alpha2 = vec![0.0; 2 * l];
    let mut linear_term = vec![0.0; 2 * l];
    let mut y = vec![0i8; 2 * l];

    let mut sum = c * param.nu * l as f64 / 2.0;
    for i in 0..l {
        let a = f64::min(sum, c);
        alpha2[i] = a;
        alpha2[i + l] = a;
        sum -= a;

        linear_term[i] = -labels[i];
        y[i] = 1;
        linear_term[i + l] = labels[i];
        y[i + l] = -1;
    }

    let q = Box::new(SvrQ::new(x, param));
    let si = Solver::solve(
        SolverVariant::Nu,
        2 * l,
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
    for i in 0..l {
        alpha[i] = alpha2[i] - alpha2[i + l];
    }

    (alpha, si)
}

// ─── svm_train_one ──────────────────────────────────────────────────

fn svm_train_one(
    x: &[Vec<SvmNode>],
    labels: &[f64],
    param: &SvmParameter,
    cp: f64,
    cn: f64,
) -> DecisionFunction {
    let (alpha, si) = match param.svm_type {
        SvmType::CSvc => solve_c_svc(x, labels, param, cp, cn),
        SvmType::NuSvc => solve_nu_svc(x, labels, param),
        SvmType::OneClass => solve_one_class(x, param),
        SvmType::EpsilonSvr => solve_epsilon_svr(x, labels, param),
        SvmType::NuSvr => solve_nu_svr(x, labels, param),
    };

    crate::info(&format!("obj = {:.6}, rho = {:.6}\n", si.obj, si.rho));

    // Count SVs
    let n_sv = alpha.iter().filter(|a| a.abs() > 0.0).count();
    let n_bsv = alpha
        .iter()
        .enumerate()
        .filter(|&(i, a)| {
            if a.abs() > 0.0 {
                if labels[i] > 0.0 {
                    a.abs() >= si.upper_bound_p
                } else {
                    a.abs() >= si.upper_bound_n
                }
            } else {
                false
            }
        })
        .count();
    crate::info(&format!("nSV = {}, nBSV = {}\n", n_sv, n_bsv));

    DecisionFunction { alpha, rho: si.rho }
}

fn mark_nonzero_indices(nonzero: &mut [bool], start: usize, alphas: &[f64]) {
    for (offset, &alpha) in alphas.iter().enumerate() {
        let idx = start + offset;
        if !nonzero[idx] && alpha.abs() > 0.0 {
            nonzero[idx] = true;
        }
    }
}

fn count_nonzero(nonzero: &[bool], start: usize, len: usize) -> usize {
    nonzero[start..start + len]
        .iter()
        .filter(|&&is_nonzero| is_nonzero)
        .count()
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
            param.gamma = 1.0 / max_index as f64;
        }
    }

    match param.svm_type {
        SvmType::OneClass | SvmType::EpsilonSvr | SvmType::NuSvr => {
            train_regression_or_one_class(problem, &param)
        }
        SvmType::CSvc | SvmType::NuSvc => train_classification(problem, &param),
    }
}

fn train_regression_or_one_class(problem: &SvmProblem, param: &SvmParameter) -> SvmModel {
    let f = svm_train_one(&problem.instances, &problem.labels, param, 0.0, 0.0);

    // Extract support vectors
    let mut sv = Vec::new();
    let mut sv_coef = Vec::new();
    let mut sv_indices = Vec::new();

    for i in 0..problem.instances.len() {
        if f.alpha[i].abs() > 0.0 {
            sv.push(problem.instances[i].clone());
            sv_coef.push(f.alpha[i]);
            sv_indices.push(i + 1); // 1-based
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

    model
}

fn train_classification(problem: &SvmProblem, param: &SvmParameter) -> SvmModel {
    let l = problem.instances.len();
    let group = group_classes(&problem.labels);
    let nr_class = group.label.len();

    if nr_class == 1 {
        crate::info("WARNING: training data in only one class. See README for details.\n");
    }

    // Reorder instances by class
    let x: Vec<&Vec<SvmNode>> = (0..l).map(|i| &problem.instances[group.perm[i]]).collect();

    // Calculate weighted C
    let mut weighted_c = vec![param.c; nr_class];
    for &(wlabel, wval) in &param.weight {
        if let Some(j) = group.label.iter().position(|&lab| lab == wlabel) {
            weighted_c[j] *= wval;
        } else {
            crate::info(&format!(
                "WARNING: class label {} specified in weight is not found\n",
                wlabel
            ));
        }
    }

    // Train k*(k-1)/2 binary classifiers
    let mut nonzero = vec![false; l];
    let n_pairs = nr_class * (nr_class - 1) / 2;
    let mut decisions = Vec::with_capacity(n_pairs);

    // Probability arrays (filled only when param.probability is set)
    let mut prob_a = Vec::new();
    let mut prob_b = Vec::new();
    if param.probability {
        prob_a.reserve(n_pairs);
        prob_b.reserve(n_pairs);
    }

    for i in 0..nr_class {
        for j in (i + 1)..nr_class {
            let si = group.start[i];
            let sj = group.start[j];
            let ci = group.count[i];
            let cj = group.count[j];

            // Build sub-problem
            let mut sub_x = Vec::with_capacity(ci + cj);
            let mut sub_labels = Vec::with_capacity(ci + cj);
            for k in 0..ci {
                sub_x.push(x[si + k].clone());
                sub_labels.push(1.0);
            }
            for k in 0..cj {
                sub_x.push(x[sj + k].clone());
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
                    weighted_c[i],
                    weighted_c[j],
                );
                prob_a.push(pa);
                prob_b.push(pb);
            }

            let f = svm_train_one(&sub_x, &sub_labels, param, weighted_c[i], weighted_c[j]);

            // Mark nonzero alphas
            mark_nonzero_indices(&mut nonzero, si, &f.alpha[..ci]);
            mark_nonzero_indices(&mut nonzero, sj, &f.alpha[ci..(ci + cj)]);

            decisions.push(f);
        }
    }

    // Build model output
    let labels: Vec<i32> = group.label.clone();
    let rho: Vec<f64> = decisions.iter().map(|d| d.rho).collect();

    // Count SVs per class
    let mut total_sv = 0;
    let mut n_sv_per_class = vec![0usize; nr_class];
    for (i, n_sv) in n_sv_per_class.iter_mut().enumerate().take(nr_class) {
        let n = count_nonzero(&nonzero, group.start[i], group.count[i]);
        total_sv += n;
        *n_sv = n;
    }

    crate::info(&format!("Total nSV = {}\n", total_sv));

    // Collect SVs and indices
    let mut model_sv = Vec::with_capacity(total_sv);
    let mut model_sv_indices = Vec::with_capacity(total_sv);
    for i in 0..l {
        if nonzero[i] {
            model_sv.push(x[i].clone());
            model_sv_indices.push(group.perm[i] + 1); // 1-based original index
        }
    }

    // Build nz_start (cumulative start of nonzero SVs per class)
    let mut nz_start = vec![0usize; nr_class];
    for i in 1..nr_class {
        nz_start[i] = nz_start[i - 1] + n_sv_per_class[i - 1];
    }

    // Build sv_coef matrix: (nr_class - 1) rows × total_sv columns
    let mut sv_coef = vec![vec![0.0; total_sv]; nr_class - 1];

    {
        let mut p = 0;
        for i in 0..nr_class {
            for j in (i + 1)..nr_class {
                let si = group.start[i];
                let sj = group.start[j];
                let ci = group.count[i];
                let cj = group.count[j];

                // Coefficients for class i's SVs go in sv_coef[j-1]
                let mut q = nz_start[i];
                for k in 0..ci {
                    if nonzero[si + k] {
                        sv_coef[j - 1][q] = decisions[p].alpha[k];
                        q += 1;
                    }
                }

                // Coefficients for class j's SVs go in sv_coef[i]
                q = nz_start[j];
                for k in 0..cj {
                    if nonzero[sj + k] {
                        sv_coef[i][q] = decisions[p].alpha[ci + k];
                        q += 1;
                    }
                }

                p += 1;
            }
        }
    }

    SvmModel {
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{load_model, load_problem};
    use crate::predict::predict;
    use std::path::PathBuf;

    fn data_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("data")
    }

    #[test]
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
        assert!(!model.sv.is_empty(), "model has no support vectors");

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
        assert!(
            (model.rho[0] - ref_model.rho[0]).abs() < 1e-4,
            "rho mismatch: Rust={}, C={}",
            model.rho[0],
            ref_model.rho[0]
        );

        // Predictions should match on training data
        let mut correct = 0;
        for (i, instance) in problem.instances.iter().enumerate() {
            let pred = predict(&model, instance);
            if pred == problem.labels[i] {
                correct += 1;
            }
        }
        let accuracy = correct as f64 / problem.labels.len() as f64;
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
                mismatches += 1;
            }
        }
        assert!(
            mismatches <= 3,
            "{} prediction mismatches between Rust-trained and C-trained models",
            mismatches
        );
    }

    #[test]
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
            if pred == problem.labels[i] {
                correct += 1;
            }
        }
        let accuracy = correct as f64 / problem.labels.len() as f64;
        assert!(
            accuracy > 0.95,
            "iris accuracy {:.2}% too low (expected >95%)",
            accuracy * 100.0
        );
    }

    #[test]
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
        assert!(!model.sv.is_empty(), "model has no support vectors");

        // Sanity-check predictions on training data.
        let mut correct = 0;
        for (i, instance) in problem.instances.iter().enumerate() {
            let pred = predict(&model, instance);
            if pred == problem.labels[i] {
                correct += 1;
            }
        }
        let accuracy = correct as f64 / problem.labels.len() as f64;
        assert!(
            accuracy > 0.70,
            "precomputed-kernel accuracy {:.2}% too low",
            accuracy * 100.0
        );
    }

    #[test]
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
        assert!(!model.sv.is_empty());
        assert_eq!(model.rho.len(), 1);

        // Predict — most training points should be classified as +1 (inlier)
        let mut inliers = 0;
        for instance in &problem.instances {
            let pred = predict(&model, instance);
            if pred > 0.0 {
                inliers += 1;
            }
        }
        let inlier_rate = inliers as f64 / problem.instances.len() as f64;
        // With nu=0.5, roughly half should be inliers (nu is upper bound on fraction of outliers)
        assert!(
            inlier_rate > 0.3 && inlier_rate < 0.9,
            "unexpected inlier rate: {:.2}%",
            inlier_rate * 100.0
        );
    }

    #[test]
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
        assert!(!model.sv.is_empty());

        // Compute MSE on training set — should be reasonable
        let mut mse = 0.0;
        for (i, instance) in problem.instances.iter().enumerate() {
            let pred = predict(&model, instance);
            let err = pred - problem.labels[i];
            mse += err * err;
        }
        mse /= problem.instances.len() as f64;

        // MSE should be finite and reasonable
        assert!(mse.is_finite(), "MSE is not finite");
        assert!(mse < 100.0, "MSE too high: {}", mse);
    }

    #[test]
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
        assert!(!model.sv.is_empty());

        let mut correct = 0;
        for (i, instance) in problem.instances.iter().enumerate() {
            let pred = predict(&model, instance);
            if pred == problem.labels[i] {
                correct += 1;
            }
        }
        let accuracy = correct as f64 / problem.labels.len() as f64;
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
        assert!(model.prob_a[0].is_finite());
        assert!(model.prob_b[0].is_finite());
    }

    #[test]
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
        assert!(!model.sv.is_empty());

        let mut mse = 0.0;
        for (i, instance) in problem.instances.iter().enumerate() {
            let pred = predict(&model, instance);
            let err = pred - problem.labels[i];
            mse += err * err;
        }
        mse /= problem.instances.len() as f64;

        assert!(mse.is_finite(), "MSE is not finite");
        assert!(mse < 200.0, "MSE too high: {}", mse);
    }
}
