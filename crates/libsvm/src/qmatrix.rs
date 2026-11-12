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

/// Fill one label-scaled kernel tail.
#[inline]
#[expect(
    clippy::cast_possible_truncation,
    reason = "C truncates the same values at these points"
)]
#[expect(
    clippy::indexing_slicing,
    reason = "the caller passes a column inside the training set and a tail inside that column"
)]
fn fill_labeled(
    data: &mut [Qfloat],
    start: usize,
    end: usize,
    i: usize,
    y: &[i8],
    kernel: &Kernel<'_>,
) {
    let yi = f64::from(y[i]);
    let rows = kernel.rows();
    let xi = rows[i];
    let x_tail = &rows[start..end];
    let y_tail = &y[start..end];
    let data_tail = &mut data[start..end];
    if let Some(sq) = kernel.squares() {
        let sq_i = sq[i];
        let sq_tail = &sq[start..end];
        for (((slot, &yj), xj), &sq_j) in data_tail.iter_mut().zip(y_tail).zip(x_tail).zip(sq_tail)
        {
            let kval = kernel.pair(xi, xj, Some(sq_i), Some(sq_j));
            *slot = (yi * f64::from(yj) * kval) as Qfloat;
        }
        return;
    }
    for ((slot, &yj), xj) in data_tail.iter_mut().zip(y_tail).zip(x_tail) {
        let kval = kernel.pair(xi, xj, None, None);
        *slot = (yi * f64::from(yj) * kval) as Qfloat;
    }
}

/// Fill one plain kernel tail.
#[inline]
#[expect(
    clippy::cast_possible_truncation,
    reason = "C truncates the same values at these points"
)]
#[expect(
    clippy::indexing_slicing,
    reason = "the caller passes a column inside the training set and a tail inside that column"
)]
fn fill_plain(data: &mut [Qfloat], start: usize, end: usize, i: usize, kernel: &Kernel<'_>) {
    let rows = kernel.rows();
    let xi = rows[i];
    let x_tail = &rows[start..end];
    let data_tail = &mut data[start..end];
    if let Some(sq) = kernel.squares() {
        let sq_i = sq[i];
        let sq_tail = &sq[start..end];
        for ((slot, xj), &sq_j) in data_tail.iter_mut().zip(x_tail).zip(sq_tail) {
            *slot = kernel.pair(xi, xj, Some(sq_i), Some(sq_j)) as Qfloat;
        }
        return;
    }
    for (slot, xj) in data_tail.iter_mut().zip(x_tail) {
        *slot = kernel.pair(xi, xj, None, None) as Qfloat;
    }
}

/// Trait for Q matrix access used by the SMO solver.
///
/// Takes `&mut self` because `get_q` mutates the internal cache.
/// The solver owns `Box<dyn QMatrix>` and copies row data into its
/// own buffers to avoid lifetime issues.
pub trait QMatrix {
    /// Get column `i` of the Q matrix, with at least `len` elements.
    ///
    /// # Panics
    ///
    /// The call panics when `i` is outside the stored column count.
    /// The call panics when `len` is longer than that count.
    fn get_q(&mut self, i: usize, len: usize) -> &[Qfloat];

    /// Get the diagonal of Q: `QD[i] = Q[i][i]`.
    fn get_qd(&self) -> &[f64];

    /// Swap indices i and j in all internal data structures.
    ///
    /// # Panics
    ///
    /// The call panics when `i` or `j` is outside the stored column count.
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
    #[must_use]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "C truncates the same values at these points"
    )]
    #[expect(
        clippy::cast_sign_loss,
        reason = "C converts the same signed values at these points"
    )]
    pub fn new(x: &'a [Vec<SvmNode>], param: &SvmParameter, y: &[i8]) -> Self {
        let l = x.len();
        let kernel = Kernel::new(x, param);
        // `cache_size` is in MB; multiply by 1048576 to convert to bytes.
        // The float→usize cast saturates on overflow; Cache::new clamps the
        // resulting byte count to a safe internal limit. check_parameter is
        // expected to have validated cache_size > 0 before we get here, but
        // even if bypassed the cast and Cache::new prevent any panic.
        let cache = Cache::new(l, (param.cache_size * 1_048_576.0) as usize);
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

impl QMatrix for SvcQ<'_> {
    /// Get column `i` of the Q matrix, with at least `len` elements.
    ///
    /// # Panics
    ///
    /// The call panics when `i` is outside the training set.
    /// The call panics when `len` is longer than the training set.
    #[expect(
        clippy::indexing_slicing,
        reason = "the cache grows the row to the requested length before this slice"
    )]
    fn get_q(&mut self, i: usize, len: usize) -> &[Qfloat] {
        let (data, start) = self.cache.get_data(i, len);
        if start < len {
            fill_labeled(data, start, len, i, &self.y, &self.kernel);
        }
        &data[..len]
    }

    fn get_qd(&self) -> &[f64] {
        &self.qd
    }

    /// Swap indices `i` and `j` in all internal data structures.
    ///
    /// # Panics
    ///
    /// The call panics when `i` or `j` is outside the training set.
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
    #[must_use]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "C truncates the same values at these points"
    )]
    #[expect(
        clippy::cast_sign_loss,
        reason = "C converts the same signed values at these points"
    )]
    pub fn new(x: &'a [Vec<SvmNode>], param: &SvmParameter) -> Self {
        let l = x.len();
        let kernel = Kernel::new(x, param);
        // See SvcQ::new for the cache_size cast note (saturating float→usize, Cache clamps).
        let cache = Cache::new(l, (param.cache_size * 1_048_576.0) as usize);
        let qd: Vec<f64> = (0..l).map(|i| kernel.evaluate(i, i)).collect();
        Self { kernel, cache, qd }
    }
}

impl QMatrix for OneClassQ<'_> {
    /// Get column `i` of the Q matrix, with at least `len` elements.
    ///
    /// # Panics
    ///
    /// The call panics when `i` is outside the training set.
    /// The call panics when `len` is longer than the training set.
    #[expect(
        clippy::indexing_slicing,
        reason = "the cache grows the row to the requested length before this slice"
    )]
    fn get_q(&mut self, i: usize, len: usize) -> &[Qfloat] {
        let (data, start) = self.cache.get_data(i, len);
        if start < len {
            fill_plain(data, start, len, i, &self.kernel);
        }
        &data[..len]
    }

    fn get_qd(&self) -> &[f64] {
        &self.qd
    }

    /// Swap indices `i` and `j` in all internal data structures.
    ///
    /// # Panics
    ///
    /// The call panics when `i` or `j` is outside the training set.
    fn swap_index(&mut self, i: usize, j: usize) {
        self.cache.swap_index(i, j);
        self.kernel.swap_index(i, j);
        self.qd.swap(i, j);
    }
}

// ─── SVR_Q ──────────────────────────────────────────────────────────

/// Q matrix for ε-SVR and ν-SVR regression.
///
/// The regression dual has 2l variables (`α_i`^+ and `α_i`^-).
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
    #[must_use]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "C truncates the same values at these points"
    )]
    #[expect(
        clippy::cast_sign_loss,
        reason = "C converts the same signed values at these points"
    )]
    #[expect(
        clippy::indexing_slicing,
        reason = "each column index stays below the doubled row count allocated above"
    )]
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "the row vector is already allocated, so twice its length and each smaller sum stay in range"
    )]
    pub fn new(x: &'a [Vec<SvmNode>], param: &SvmParameter) -> Self {
        let l = x.len();
        let kernel = Kernel::new(x, param);
        // See SvcQ::new for the cache_size cast note (saturating float→usize, Cache clamps).
        let cache = Cache::new(l, (param.cache_size * 1_048_576.0) as usize);

        let l2 = l * 2;
        let mut sign = vec![0i8; l2];
        let mut index = vec![0usize; l2];
        let mut qd = vec![0.0f64; l2];

        for k in 0..l {
            let hi = k + l;
            sign[k] = 1;
            sign[hi] = -1;
            index[k] = k;
            index[hi] = k;
            let kk = kernel.evaluate(k, k);
            qd[k] = kk;
            qd[hi] = kk;
        }

        let buffer = [vec![0.0; l2], vec![0.0; l2]];

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

impl QMatrix for SvrQ<'_> {
    /// Get column `i` of the Q matrix, with at least `len` elements.
    ///
    /// # Panics
    ///
    /// The call panics when `i` is outside the doubled training set.
    /// The call panics when `len` is longer than that set.
    #[expect(
        clippy::indexing_slicing,
        reason = "the caller stays inside the doubled training set, and each stored index stays inside the kernel row"
    )]
    fn get_q(&mut self, i: usize, len: usize) -> &[Qfloat] {
        let real_i = self.index[i];
        let n = self.l;

        // Fetch (or fill) the full kernel row for the real data index.
        let (data, start) = self.cache.get_data(real_i, n);
        if start < n {
            fill_plain(data, start, n, real_i, &self.kernel);
        }

        // Each stored index stays below `n`.
        // `new` writes `k`, and swap only exchanges those values.
        let buf_idx = self.next_buffer;
        self.next_buffer ^= 1;
        let si = f32::from(self.sign[i]);
        let sign = &self.sign[..len];
        let index = &self.index[..len];
        let buf = &mut self.buffer[buf_idx][..len];
        for (slot, (&sign_j, &idx)) in buf.iter_mut().zip(sign.iter().zip(index)) {
            *slot = si * f32::from(sign_j) * data[idx];
        }
        &self.buffer[buf_idx][..len]
    }

    fn get_qd(&self) -> &[f64] {
        &self.qd
    }

    /// Swap indices `i` and `j` in all internal data structures.
    ///
    /// # Panics
    ///
    /// The call panics when `i` or `j` is outside the doubled training set.
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

    fn copy_prefix(dest: &mut [f32], src: &[f32], n: usize) {
        let Some(head) = dest.get_mut(..n) else {
            assert!(dest.len() >= n);
            return;
        };
        let Some(row) = src.get(..n) else {
            assert!(src.len() >= n);
            return;
        };
        head.copy_from_slice(row);
    }

    fn entry(matrix: &[Vec<f32>], i: usize, j: usize) -> f32 {
        let Some(row) = matrix.get(i) else {
            assert!(matrix.len() > i);
            return 0.0;
        };
        let Some(&value) = row.get(j) else {
            assert!(row.len() > j);
            return 0.0;
        };
        value
    }

    fn at(row: &[f32], j: usize) -> f32 {
        let Some(&value) = row.get(j) else {
            assert!(row.len() > j);
            return 0.0;
        };
        value
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
            let Some(dest) = matrix.get_mut(i) else {
                assert!(matrix.len() > i);
                return;
            };
            copy_prefix(dest, &row, l);
        }

        // Check symmetry
        for i in 0..l {
            for j in 0..l {
                let left = entry(&matrix, i, j);
                let right = entry(&matrix, j, i);
                assert!(
                    (left - right).abs() < 1e-6,
                    "Q[{i},{j}]={left} != Q[{j},{i}]={right}"
                );
            }
        }

        // Check sign: Q[0][1] should be negative (y[0]*y[1] = -1)
        assert!(entry(&matrix, 0, 1) < 0.0);
        // Q[0][2] should be positive (y[0]*y[2] = +1)
        assert!(entry(&matrix, 0, 2) > 0.0);
    }

    #[test]
    fn one_class_q_no_sign_scaling() {
        let data = vec![make_nodes(&[(1, 1.0)]), make_nodes(&[(1, 2.0)])];
        let param = default_rbf_param();
        let mut q = OneClassQ::new(&data, &param);

        let row = q.get_q(0, 2);
        // All values should be positive (kernel values are always positive for RBF)
        assert!(at(row, 0) > 0.0);
        assert!(at(row, 1) > 0.0);
        // Diagonal should be 1.0
        assert!((at(row, 0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn svr_q_double_buffer() {
        let data = vec![make_nodes(&[(1, 1.0)]), make_nodes(&[(1, 2.0)])];
        let param = default_rbf_param();
        let mut q = SvrQ::new(&data, &param);
        let l2 = data.len().saturating_mul(2); // 4

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
        let q02 = at(&row0, 2);
        assert!((q02 - (-1.0)).abs() < 1e-6, "Q[0][2] = {q02}");
    }
}
