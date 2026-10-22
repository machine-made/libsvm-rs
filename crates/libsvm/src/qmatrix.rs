//! Q matrix implementations for the SMO solver.
//!
//! The Q matrix encodes the quadratic form in the SVM dual problem:
//! `Q[i][j] = y[i] * y[j] * K(x[i], x[j])` for classification,
//! or just `K(i,j)` for one-class, or a signed/indexed variant for SVR.
//!
//! Each implementation wraps a `Kernel` and a `Cache`, providing the
//! `QMatrix` trait that the solver consumes.

use crate::cache::{Cache, Qfloat};
use crate::kernel::Kernel;
use crate::types::{SvmNode, SvmParameter};

/// Trait for Q matrix access used by the SMO solver.
///
/// Takes `&mut self` because `get_q` mutates the internal cache.
/// The solver owns `Box<dyn QMatrix>` and copies row data into its
/// own buffers to avoid lifetime issues.
pub trait QMatrix {
    /// Get column `i` of the Q matrix, with at least `len` elements.
    fn get_q(&mut self, i: usize, len: usize) -> &[Qfloat];

    /// Get the diagonal of Q: `QD[i] = Q[i][i]`.
    fn get_qd(&self) -> &[f64];

    /// Swap indices i and j in all internal data structures.
    fn swap_index(&mut self, i: usize, j: usize);
}

// ─── SVC_Q ──────────────────────────────────────────────────────────

/// Q matrix for C-SVC and ν-SVC classification.
///
/// `Q[i][j] = y[i] * y[j] * K(x[i], x[j])` stored as `Qfloat` (f32).
pub struct SvcQ<'a> {
    kernel: Kernel<'a>,
    cache: Cache,
    y: Vec<i8>,
    qd: Vec<f64>,
}

impl<'a> SvcQ<'a> {
    /// Build the classification Q matrix used by LIBSVM's `SVC_Q` helper.
    pub fn new(x: &'a [Vec<SvmNode>], param: &SvmParameter, y: &[i8]) -> Self {
        let l = x.len();
        let kernel = Kernel::new(x, param);
        // `cache_size` is in MB; multiply by 1048576 to convert to bytes.
        // The float→usize cast saturates on overflow; Cache::new clamps the
        // resulting byte count to a safe internal limit. check_parameter is
        // expected to have validated cache_size > 0 before we get here, but
        // even if bypassed the cast and Cache::new prevent any panic.
        let cache = Cache::new(l, (param.cache_size * 1048576.0) as usize);
        let qd: Vec<f64> = (0..l).map(|i| kernel.evaluate(i, i)).collect();
        let y = y.to_vec();
        Self {
            kernel,
            cache,
            y,
            qd,
        }
    }
}

#[allow(clippy::needless_range_loop)]
impl<'a> QMatrix for SvcQ<'a> {
    fn get_q(&mut self, i: usize, len: usize) -> &[Qfloat] {
        let (data, start) = self.cache.get_data(i, len);
        if start < len {
            let yi = self.y[i] as f64;
            for j in start..len {
                let kval = self.kernel.evaluate(i, j);
                data[j] = (yi * self.y[j] as f64 * kval) as Qfloat;
            }
        }
        &data[..len]
    }

    fn get_qd(&self) -> &[f64] {
        &self.qd
    }

    fn swap_index(&mut self, i: usize, j: usize) {
        self.cache.swap_index(i, j);
        self.kernel.swap_index(i, j);
        self.y.swap(i, j);
        self.qd.swap(i, j);
    }
}

// ─── ONE_CLASS_Q ────────────────────────────────────────────────────

/// Q matrix for one-class SVM.
///
/// `Q[i][j] = K(x[i], x[j])` — no label scaling.
pub struct OneClassQ<'a> {
    kernel: Kernel<'a>,
    cache: Cache,
    qd: Vec<f64>,
}

impl<'a> OneClassQ<'a> {
    /// Build the one-class Q matrix used by LIBSVM's `ONE_CLASS_Q` helper.
    pub fn new(x: &'a [Vec<SvmNode>], param: &SvmParameter) -> Self {
        let l = x.len();
        let kernel = Kernel::new(x, param);
        // See SvcQ::new for the cache_size cast note (saturating float→usize, Cache clamps).
        let cache = Cache::new(l, (param.cache_size * 1048576.0) as usize);
        let qd: Vec<f64> = (0..l).map(|i| kernel.evaluate(i, i)).collect();
        Self { kernel, cache, qd }
    }
}

#[allow(clippy::needless_range_loop)]
impl<'a> QMatrix for OneClassQ<'a> {
    fn get_q(&mut self, i: usize, len: usize) -> &[Qfloat] {
        let (data, start) = self.cache.get_data(i, len);
        if start < len {
            for j in start..len {
                data[j] = self.kernel.evaluate(i, j) as Qfloat;
            }
        }
        &data[..len]
    }

    fn get_qd(&self) -> &[f64] {
        &self.qd
    }

    fn swap_index(&mut self, i: usize, j: usize) {
        self.cache.swap_index(i, j);
        self.kernel.swap_index(i, j);
        self.qd.swap(i, j);
    }
}

// ─── SVR_Q ──────────────────────────────────────────────────────────

/// Q matrix for ε-SVR and ν-SVR regression.
///
/// The regression dual has 2l variables (α_i^+ and α_i^-).
/// The underlying kernel cache stores only l rows of actual kernel
/// evaluations; `get_q` reorders/signs them into a double-buffered output.
pub struct SvrQ<'a> {
    kernel: Kernel<'a>,
    cache: Cache,
    /// Number of original data points.
    l: usize,
    /// Sign of each of the 2l variables: +1 for first l, -1 for second l.
    sign: Vec<i8>,
    /// Maps each of the 2l indices to the original data index [0..l).
    index: Vec<usize>,
    /// Diagonal of the 2l×2l Q matrix.
    qd: Vec<f64>,
    /// Double buffer for returning Q rows (solver may hold two simultaneously).
    buffer: [Vec<Qfloat>; 2],
    /// Toggle between the two buffers.
    next_buffer: usize,
}

impl<'a> SvrQ<'a> {
    /// Build the regression Q matrix used by LIBSVM's `SVR_Q` helper.
    pub fn new(x: &'a [Vec<SvmNode>], param: &SvmParameter) -> Self {
        let l = x.len();
        let kernel = Kernel::new(x, param);
        // See SvcQ::new for the cache_size cast note (saturating float→usize, Cache clamps).
        let cache = Cache::new(l, (param.cache_size * 1048576.0) as usize);

        let mut sign = vec![0i8; 2 * l];
        let mut index = vec![0usize; 2 * l];
        let mut qd = vec![0.0f64; 2 * l];

        for k in 0..l {
            sign[k] = 1;
            sign[k + l] = -1;
            index[k] = k;
            index[k + l] = k;
            let kk = kernel.evaluate(k, k);
            qd[k] = kk;
            qd[k + l] = kk;
        }

        let buffer = [vec![0.0 as Qfloat; 2 * l], vec![0.0 as Qfloat; 2 * l]];

        Self {
            kernel,
            cache,
            l,
            sign,
            index,
            qd,
            buffer,
            next_buffer: 0,
        }
    }
}

#[allow(clippy::needless_range_loop)]
impl<'a> QMatrix for SvrQ<'a> {
    fn get_q(&mut self, i: usize, len: usize) -> &[Qfloat] {
        let real_i = self.index[i];
        let l = self.l;

        // Fetch (or fill) the full kernel row for the real data index
        let (data, start) = self.cache.get_data(real_i, l);
        if start < l {
            for j in start..l {
                data[j] = self.kernel.evaluate(real_i, j) as Qfloat;
            }
        }

        // Reorder and apply signs into the output buffer.
        // Slice buf/sign/index to `len` so the compiler drops the
        // per-element bounds checks; `data` is an l-length cache row and
        // `index[j] < l` is a data-dependent lookup that stays checked.
        let buf_idx = self.next_buffer;
        self.next_buffer = 1 - self.next_buffer;
        let si = self.sign[i] as f32;
        let buf = &mut self.buffer[buf_idx][..len];
        let sign = &self.sign[..len];
        let index = &self.index[..len];
        for j in 0..len {
            buf[j] = si * (sign[j] as f32) * data[index[j]];
        }
        &self.buffer[buf_idx][..len]
    }

    fn get_qd(&self) -> &[f64] {
        &self.qd
    }

    fn swap_index(&mut self, i: usize, j: usize) {
        self.sign.swap(i, j);
        self.index.swap(i, j);
        self.qd.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{KernelType, SvmNode, SvmParameter};

    fn make_nodes(pairs: &[(i32, f64)]) -> Vec<SvmNode> {
        pairs
            .iter()
            .map(|&(i, v)| SvmNode { index: i, value: v })
            .collect()
    }

    fn default_rbf_param() -> SvmParameter {
        SvmParameter {
            kernel_type: KernelType::Rbf,
            gamma: 0.5,
            cache_size: 1.0,
            ..Default::default()
        }
    }

    #[test]
    fn svc_q_diagonal_equals_one_for_rbf() {
        let data = vec![
            make_nodes(&[(1, 1.0), (2, 0.0)]),
            make_nodes(&[(1, 0.0), (2, 1.0)]),
        ];
        let y = vec![1i8, -1i8];
        let param = default_rbf_param();
        let q = SvcQ::new(&data, &param, &y);
        // K(x,x) = 1 for RBF, QD[i] = y[i]*y[i]*1 = 1
        for &d in q.get_qd() {
            assert!((d - 1.0).abs() < 1e-15);
        }
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn svc_q_symmetry_and_sign() {
        let data = vec![
            make_nodes(&[(1, 1.0)]),
            make_nodes(&[(1, 2.0)]),
            make_nodes(&[(1, 3.0)]),
        ];
        let y = vec![1i8, -1i8, 1i8];
        let param = default_rbf_param();
        let mut q = SvcQ::new(&data, &param, &y);
        let l = data.len();

        // Collect full matrix
        let mut matrix = vec![vec![0.0f32; l]; l];
        for i in 0..l {
            let row = q.get_q(i, l).to_vec();
            matrix[i][..l].copy_from_slice(&row[..l]);
        }

        // Check symmetry
        for i in 0..l {
            for j in 0..l {
                assert!(
                    (matrix[i][j] - matrix[j][i]).abs() < 1e-6,
                    "Q[{},{}]={} != Q[{},{}]={}",
                    i,
                    j,
                    matrix[i][j],
                    j,
                    i,
                    matrix[j][i]
                );
            }
        }

        // Check sign: Q[0][1] should be negative (y[0]*y[1] = -1)
        assert!(matrix[0][1] < 0.0);
        // Q[0][2] should be positive (y[0]*y[2] = +1)
        assert!(matrix[0][2] > 0.0);
    }

    #[test]
    fn one_class_q_no_sign_scaling() {
        let data = vec![make_nodes(&[(1, 1.0)]), make_nodes(&[(1, 2.0)])];
        let param = default_rbf_param();
        let mut q = OneClassQ::new(&data, &param);

        let row = q.get_q(0, 2);
        // All values should be positive (kernel values are always positive for RBF)
        assert!(row[0] > 0.0);
        assert!(row[1] > 0.0);
        // Diagonal should be 1.0
        assert!((row[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn svr_q_double_buffer() {
        let data = vec![make_nodes(&[(1, 1.0)]), make_nodes(&[(1, 2.0)])];
        let param = default_rbf_param();
        let mut q = SvrQ::new(&data, &param);
        let l2 = 2 * data.len(); // 4

        // Get two rows — they use different buffers
        let row0 = q.get_q(0, l2).to_vec();
        let row1 = q.get_q(1, l2).to_vec();

        // Row 0: sign[0]=+1, index[0]=0 → K(0, index[j]) * sign[0] * sign[j]
        // Row 1: sign[1]=+1, index[1]=1 → K(1, index[j]) * sign[1] * sign[j]
        // Both should have non-zero entries
        assert!(row0.iter().any(|&v| v != 0.0));
        assert!(row1.iter().any(|&v| v != 0.0));

        // For index 0 (sign +1) and index 2 (sign -1, real_idx 0):
        // Q[0][2] = sign[0]*sign[2]*K(0,0) = 1*(-1)*1 = -1
        assert!((row0[2] - (-1.0)).abs() < 1e-6, "Q[0][2] = {}", row0[2]);
    }
}
