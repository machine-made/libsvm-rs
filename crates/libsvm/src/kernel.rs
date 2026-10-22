//! Kernel functions matching the original LIBSVM.
//!
//! Provides both:
//! - Standalone `k_function` for prediction (operates on sparse node slices)
//! - `Kernel` struct for training (precomputes x_square for RBF, stores refs)

use crate::types::{KernelType, SvmNode, SvmParameter};

// ─── Integer power (matches LIBSVM's powi) ─────────────────────────

/// Integer power by squaring. Matches LIBSVM's `powi(base, times)`.
///
/// For negative `times`, returns 1.0 (same as the C code, which only
/// iterates while `t > 0`).
#[inline]
pub fn powi(base: f64, times: i32) -> f64 {
    let mut tmp = base;
    let mut ret = 1.0;
    let mut t = times;
    while t > 0 {
        if t % 2 == 1 {
            ret *= tmp;
        }
        tmp *= tmp;
        t /= 2;
    }
    ret
}

// ─── Sparse dot product ─────────────────────────────────────────────

/// Sparse dot product of two sorted-by-index node slices.
///
/// This is the merge-based O(n+m) algorithm from LIBSVM.
#[inline]
pub fn dot(x: &[SvmNode], y: &[SvmNode]) -> f64 {
    let mut sum = 0.0;
    let mut ix = 0;
    let mut iy = 0;
    let x_len = x.len();
    let y_len = y.len();
    while ix < x_len && iy < y_len {
        let x_node = &x[ix];
        let y_node = &y[iy];
        if x_node.index == y_node.index {
            sum += x_node.value * y_node.value;
            ix += 1;
            iy += 1;
        } else if x_node.index > y_node.index {
            iy += 1;
        } else {
            ix += 1;
        }
    }
    sum
}

/// Squared Euclidean distance for sparse vectors (used by RBF k_function).
///
/// Computes ‖x - y‖² without computing the difference vector.
#[inline]
fn sparse_sq_dist(x: &[SvmNode], y: &[SvmNode]) -> f64 {
    let mut sum = 0.0;
    let mut ix = 0;
    let mut iy = 0;
    while ix < x.len() && iy < y.len() {
        if x[ix].index == y[iy].index {
            let d = x[ix].value - y[iy].value;
            sum += d * d;
            ix += 1;
            iy += 1;
        } else if x[ix].index > y[iy].index {
            sum += y[iy].value * y[iy].value;
            iy += 1;
        } else {
            sum += x[ix].value * x[ix].value;
            ix += 1;
        }
    }
    // Drain remaining elements
    while ix < x.len() {
        sum += x[ix].value * x[ix].value;
        ix += 1;
    }
    while iy < y.len() {
        sum += y[iy].value * y[iy].value;
        iy += 1;
    }
    sum
}

// ─── Standalone kernel evaluation ───────────────────────────────────

/// Evaluate the kernel function K(x, y) for the given parameters.
///
/// This is the standalone version used during prediction. Matches
/// LIBSVM's `Kernel::k_function`.
pub fn k_function(x: &[SvmNode], y: &[SvmNode], param: &SvmParameter) -> f64 {
    match param.kernel_type {
        KernelType::Linear => dot(x, y),
        KernelType::Polynomial => powi(param.gamma * dot(x, y) + param.coef0, param.degree),
        KernelType::Rbf => (-param.gamma * sparse_sq_dist(x, y)).exp(),
        KernelType::Sigmoid => (param.gamma * dot(x, y) + param.coef0).tanh(),
        KernelType::Precomputed => {
            // For precomputed kernels, x[y[0].value as index] gives the value.
            // y[0].value is the column index (1-based SV index).
            //
            // Safety note: `node.value as usize` is a float→usize cast that saturates
            // on out-of-range values (negative or > usize::MAX) — this matches upstream
            // C++ behaviour. `.get()` then maps any out-of-range index to None, and
            // `.map_or(0.0, ...)` returns 0.0 in that case (silent mis-map, no panic).
            // Intentional: check_parameter validates the precomputed kernel at train time;
            // at predict time the model loader validates the 0:serial_number row header.
            y.first()
                .and_then(|node| x.get(node.value as usize))
                .map_or(0.0, |n| n.value)
        }
    }
}

// ─── Kernel struct for training ─────────────────────────────────────

/// Kernel evaluator for training. Holds references to the dataset and
/// precomputes `x_square[i] = dot(x[i], x[i])` for RBF kernels.
///
/// Stores `Vec<&'a [SvmNode]>` so that the solver can swap entries
/// during shrinking (mirroring the C++ pointer-array swap trick).
///
/// The `kernel_function` method pointer pattern from C++ is replaced
/// by a match on `kernel_type` — the branch predictor handles this
/// efficiently since the type doesn't change during training.
pub struct Kernel<'a> {
    x: Vec<&'a [SvmNode]>,
    x_square: Option<Vec<f64>>,
    kernel_type: KernelType,
    degree: i32,
    gamma: f64,
    coef0: f64,
}

impl<'a> Kernel<'a> {
    /// Create a new kernel evaluator for the given dataset and parameters.
    pub fn new(x: &'a [Vec<SvmNode>], param: &SvmParameter) -> Self {
        let x_refs: Vec<&'a [SvmNode]> = x.iter().map(|xi| xi.as_slice()).collect();
        let x_square = if param.kernel_type == KernelType::Rbf {
            Some(x_refs.iter().map(|xi| dot(xi, xi)).collect())
        } else {
            None
        };

        Self {
            x: x_refs,
            x_square,
            kernel_type: param.kernel_type,
            degree: param.degree,
            gamma: param.gamma,
            coef0: param.coef0,
        }
    }

    /// Evaluate K(x\[i\], x\[j\]) using precomputed data where possible.
    #[inline]
    pub fn evaluate(&self, i: usize, j: usize) -> f64 {
        let xi = self.x[i];
        let xj = self.x[j];
        match self.kernel_type {
            KernelType::Linear => dot(xi, xj),
            KernelType::Polynomial => powi(self.gamma * dot(xi, xj) + self.coef0, self.degree),
            KernelType::Rbf => {
                // Use precomputed x_square: ‖x_i - x_j‖² = x_sq[i] + x_sq[j] - 2*dot(x_i, x_j)
                let val = if let Some(sq) = &self.x_square {
                    sq[i] + sq[j] - 2.0 * dot(xi, xj)
                } else {
                    sparse_sq_dist(xi, xj)
                };
                (-self.gamma * val).exp()
            }
            KernelType::Sigmoid => (self.gamma * dot(xi, xj) + self.coef0).tanh(),
            // See the free-function kernel_function above for the safety note on the
            // float→usize cast: saturates on out-of-range values, .get() maps to None,
            // .map_or returns 0.0 (intentional, matches upstream; silent mis-map, no panic).
            KernelType::Precomputed => xj
                .first()
                .and_then(|node| xi.get(node.value as usize))
                .map_or(0.0, |n| n.value),
        }
    }

    /// Swap data-point references and precomputed squares at positions i and j.
    ///
    /// Used by QMatrix implementations during solver shrinking.
    pub fn swap_index(&mut self, i: usize, j: usize) {
        self.x.swap(i, j);
        if let Some(ref mut sq) = self.x_square {
            sq.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SvmParameter;

    fn make_nodes(pairs: &[(i32, f64)]) -> Vec<SvmNode> {
        pairs
            .iter()
            .map(|&(index, value)| SvmNode { index, value })
            .collect()
    }

    #[test]
    fn powi_basic() {
        assert_eq!(powi(2.0, 10), 1024.0);
        assert_eq!(powi(3.0, 0), 1.0);
        assert_eq!(powi(5.0, 1), 5.0);
        assert!((powi(2.0, 3) - 8.0).abs() < 1e-15);
        // Negative exponent: LIBSVM returns 1.0 (loop doesn't execute)
        assert_eq!(powi(2.0, -1), 1.0);
    }

    #[test]
    fn dot_product() {
        let x = make_nodes(&[(1, 1.0), (3, 2.0), (5, 3.0)]);
        let y = make_nodes(&[(1, 4.0), (2, 5.0), (5, 6.0)]);
        // dot = 1*4 + 3*6 = 4 + 18 = 22
        assert!((dot(&x, &y) - 22.0).abs() < 1e-15);
    }

    #[test]
    fn dot_disjoint() {
        let x = make_nodes(&[(1, 1.0), (3, 2.0)]);
        let y = make_nodes(&[(2, 5.0), (4, 6.0)]);
        assert_eq!(dot(&x, &y), 0.0);
    }

    #[test]
    fn dot_empty() {
        let x = make_nodes(&[]);
        let y = make_nodes(&[(1, 1.0)]);
        assert_eq!(dot(&x, &y), 0.0);
    }

    #[test]
    fn kernel_linear() {
        let x = make_nodes(&[(1, 1.0), (2, 2.0)]);
        let y = make_nodes(&[(1, 3.0), (2, 4.0)]);
        let param = SvmParameter {
            kernel_type: KernelType::Linear,
            ..Default::default()
        };
        assert!((k_function(&x, &y, &param) - 11.0).abs() < 1e-15);
    }

    #[test]
    fn kernel_rbf() {
        let x = make_nodes(&[(1, 1.0), (2, 0.0)]);
        let y = make_nodes(&[(1, 0.0), (2, 1.0)]);
        let param = SvmParameter {
            kernel_type: KernelType::Rbf,
            gamma: 0.5,
            ..Default::default()
        };
        // ‖x-y‖² = 1+1 = 2, K = exp(-0.5 * 2) = exp(-1)
        let expected = (-1.0_f64).exp();
        assert!((k_function(&x, &y, &param) - expected).abs() < 1e-15);
    }

    #[test]
    fn kernel_poly() {
        let x = make_nodes(&[(1, 1.0), (2, 2.0)]);
        let y = make_nodes(&[(1, 3.0), (2, 4.0)]);
        let param = SvmParameter {
            kernel_type: KernelType::Polynomial,
            gamma: 1.0,
            coef0: 1.0,
            degree: 2,
            ..Default::default()
        };
        // (1*1*11 + 1)^2 = 12^2 = 144
        assert!((k_function(&x, &y, &param) - 144.0).abs() < 1e-15);
    }

    #[test]
    fn kernel_sigmoid() {
        let x = make_nodes(&[(1, 1.0)]);
        let y = make_nodes(&[(1, 1.0)]);
        let param = SvmParameter {
            kernel_type: KernelType::Sigmoid,
            gamma: 1.0,
            coef0: 0.0,
            ..Default::default()
        };
        // tanh(1*1 + 0) = tanh(1)
        let expected = 1.0_f64.tanh();
        assert!((k_function(&x, &y, &param) - expected).abs() < 1e-15);
    }

    #[test]
    fn kernel_precomputed() {
        // Precomputed rows use 0:sample_id followed by 1..l kernel values.
        let x = make_nodes(&[(0, 1.0), (1, 1.5), (2, 2.5)]);
        let y = make_nodes(&[(0, 2.0), (1, 1.5), (2, 2.5)]);
        let param = SvmParameter {
            kernel_type: KernelType::Precomputed,
            ..Default::default()
        };

        // y[0].value = 2 => take x[2] => 2.5
        assert!((k_function(&x, &y, &param) - 2.5).abs() < 1e-15);

        let data = vec![x.clone(), y.clone()];
        let kern = Kernel::new(&data, &param);
        // evaluate(0,1) uses column index from row 1 sample id (2)
        assert!((kern.evaluate(0, 1) - 2.5).abs() < 1e-15);
    }

    #[test]
    fn kernel_struct_matches_standalone() {
        let data = vec![
            make_nodes(&[(1, 0.5), (3, -1.0)]),
            make_nodes(&[(1, -0.25), (2, 0.75)]),
            make_nodes(&[(2, 1.0), (3, 0.5)]),
        ];
        let param = SvmParameter {
            kernel_type: KernelType::Rbf,
            gamma: 0.5,
            ..Default::default()
        };

        let kern = Kernel::new(&data, &param);

        // Verify Kernel::evaluate matches k_function for all pairs
        for i in 0..data.len() {
            for j in 0..data.len() {
                let via_struct = kern.evaluate(i, j);
                let via_func = k_function(&data[i], &data[j], &param);
                assert!(
                    (via_struct - via_func).abs() < 1e-15,
                    "mismatch at ({},{}): {} vs {}",
                    i,
                    j,
                    via_struct,
                    via_func
                );
            }
        }
    }

    #[test]
    fn rbf_self_kernel_is_one() {
        let x = make_nodes(&[(1, 3.0), (5, -2.0), (10, 0.7)]);
        let param = SvmParameter {
            kernel_type: KernelType::Rbf,
            gamma: 1.0,
            ..Default::default()
        };
        // K(x, x) = exp(-γ * 0) = 1
        assert!((k_function(&x, &x, &param) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn precomputed_kernel_missing_sample_serial_number_returns_zero() {
        let x = make_nodes(&[(0, 1.0), (1, 2.0)]);
        let y = Vec::new();
        let param = SvmParameter {
            kernel_type: KernelType::Precomputed,
            ..Default::default()
        };
        assert_eq!(k_function(&x, &y, &param), 0.0);
    }
}
