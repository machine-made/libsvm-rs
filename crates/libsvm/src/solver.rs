//! SMO solver for the SVM dual problem.
//!
//! Implements the Fan et al. (JMLR 2005) algorithm with WSS3 working-set
//! selection, shrinking heuristic, and both Standard and Nu variants.
//!
//! This is a faithful translation of `Solver` and `Solver_NU` from
//! LIBSVM's `svm.cpp` (lines 362–1265).

use crate::qmatrix::QMatrix;

const TAU: f64 = 1e-12;
const INF: f64 = f64::INFINITY;

/// Raised when a solver vector does not cover an index.
struct RowFault {
    name: &'static str,
    index: usize,
    len: usize,
}

/// Return one prefix of length `n`.
#[expect(
    clippy::unnecessary_lazy_evaluations,
    reason = "the row fault is built only when the prefix is missing"
)]
fn cut<'a, T>(items: &'a [T], n: usize, name: &'static str) -> Result<&'a [T], RowFault> {
    let len = items.len();
    items.get(..n).ok_or_else(|| RowFault {
        name,
        index: n,
        len,
    })
}

/// Return one copied element at `index`.
#[expect(
    clippy::unnecessary_lazy_evaluations,
    reason = "the row fault is built only when the index is missing"
)]
fn take<T: Copy>(items: &[T], index: usize, name: &'static str) -> Result<T, RowFault> {
    let len = items.len();
    items
        .get(index)
        .copied()
        .ok_or_else(|| RowFault { name, index, len })
}

/// Return one mutable element at `index`.
#[expect(
    clippy::unnecessary_lazy_evaluations,
    reason = "the row fault is built only when the index is missing"
)]
fn slot<'a, T>(
    items: &'a mut [T],
    index: usize,
    name: &'static str,
) -> Result<&'a mut T, RowFault> {
    let len = items.len();
    items
        .get_mut(index)
        .ok_or_else(|| RowFault { name, index, len })
}

/// Largest row count for which every smaller count converts exactly to `f64`: 2^53.
const MAX_EXACT_ROWS: u64 = 9_007_199_254_740_992;

/// Result of the solver.
#[derive(Debug, Clone)]
pub struct SolutionInfo {
    /// Final dual objective value reported like LIBSVM's `SolutionInfo::obj`.
    pub obj: f64,
    /// Bias term (`rho`) for the solved decision function.
    pub rho: f64,
    /// Positive-class box constraint used by the solved sub-problem.
    pub upper_bound_p: f64,
    /// Negative-class box constraint used by the solved sub-problem.
    pub upper_bound_n: f64,
    /// Extra value for Nu solver: `(r1 + r2) / 2`.
    pub r: f64,
}

/// Alpha variable status relative to its box constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AlphaStatus {
    LowerBound,
    UpperBound,
    Free,
}

/// Standard vs Nu solver variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverVariant {
    /// Standard LIBSVM `Solver` variant for C-SVC, one-class SVM, and ε-SVR.
    Standard,
    /// LIBSVM `Solver_NU` variant for ν-SVC and ν-SVR.
    Nu,
}

/// SMO solver.
pub struct Solver<'a> {
    l: usize,
    active_size: usize,
    variant: SolverVariant,

    y: Vec<i8>,
    g: Vec<f64>,
    g_bar: Vec<f64>,
    alpha: Vec<f64>,
    alpha_status: Vec<AlphaStatus>,
    p: Vec<f64>,
    active_set: Vec<usize>,
    unshrink: bool,

    q: Box<dyn QMatrix + 'a>,
    qd: Vec<f64>,
    cp: f64,
    cn: f64,
    eps: f64,
}

impl<'a> Solver<'a> {
    /// Run the SMO solver.
    ///
    /// # Arguments
    /// * `variant` — Standard or Nu
    /// * `l` — problem size
    /// * `q` — Q matrix (ownership transferred)
    /// * `p_` — linear term
    /// * `y_` — labels (+1/-1)
    /// * `alpha_` — initial alpha (modified in place with solution)
    /// * `cp`, `cn` — box constraints for positive/negative classes
    /// * `eps` — stopping tolerance
    /// * `shrinking` — whether to use the shrinking heuristic
    ///
    /// # Panics
    ///
    /// The call panics when `p_`, `y_`, `alpha_`, or the Q diagonal is shorter than `l`.
    /// The call panics when `l` is above 2^53, the largest row count whose free-row counts convert exactly to `f64`.
    /// The call panics when a solver vector is shorter than its index.
    /// The call panics when `alpha_` does not cover an active-set index.
    /// The call panics when a Q row is shorter than the requested length.
    /// The call panics when a Q column index is outside the stored column count.
    #[expect(
        clippy::too_many_arguments,
        reason = "Keep the C solver arguments in one call"
    )]
    #[expect(
        clippy::expect_used,
        reason = "run checks that p, y, alpha and the Q diagonal cover l; each Q row has the length the QMatrix contract promises"
    )]
    pub fn solve(
        variant: SolverVariant,
        l: usize,
        q: Box<dyn QMatrix + 'a>,
        p_: &[f64],
        y_: &[i8],
        alpha_: &mut [f64],
        cp: f64,
        cn: f64,
        eps: f64,
        shrinking: bool,
    ) -> SolutionInfo {
        Self::run(variant, l, q, p_, y_, alpha_, cp, cn, eps, shrinking)
            .map_err(|RowFault { name, index, len }| {
                if name == "l" {
                    format!(
                        "the row count {index} is above 2^53, the largest count that converts exactly to f64; solve fewer rows"
                    )
                } else {
                    format!("solver row {name} does not cover index {index} of length {len}")
                }
            })
            .expect("each solver vector covers its index")
    }

    /// Run one solve and return a row fault instead of panicking.
    #[expect(
        clippy::too_many_arguments,
        reason = "Keep the C solver arguments in one call"
    )]
    #[expect(
        clippy::many_single_char_names,
        reason = "The C solver uses these short names"
    )]
    #[expect(
        clippy::single_match_else,
        reason = "Keep the C branch order for comparison with the original"
    )]
    #[expect(clippy::too_many_lines, reason = "Keep the source loop in one place")]
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "the scaled limit, the counter and the iteration stay inside usize"
    )]
    fn run(
        variant: SolverVariant,
        l: usize,
        q: Box<dyn QMatrix + 'a>,
        p_: &[f64],
        y_: &[i8],
        alpha_: &mut [f64],
        cp: f64,
        cn: f64,
        eps: f64,
        shrinking: bool,
    ) -> Result<SolutionInfo, RowFault> {
        // Every count of rows up to 2^53 converts exactly to f64 (see the rho code).
        if !matches!(u64::try_from(l), Ok(rows) if rows <= MAX_EXACT_ROWS) {
            return Err(RowFault {
                name: "l",
                index: l,
                len: usize::MAX,
            });
        }
        let qd = cut(q.get_qd(), l, "qd")?.to_vec();
        let p = cut(p_, l, "p")?.to_vec();
        let y = cut(y_, l, "y")?.to_vec();
        let alpha = cut(alpha_, l, "alpha")?.to_vec();

        let mut solver = Solver {
            l,
            active_size: l,
            variant,
            y,
            g: vec![0.0; l],
            g_bar: vec![0.0; l],
            alpha,
            alpha_status: vec![AlphaStatus::LowerBound; l],
            p,
            active_set: (0..l).collect(),
            unshrink: false,
            q,
            qd,
            cp,
            cn,
            eps,
        };

        // Initialize alpha_status
        for i in 0..l {
            solver.update_alpha_status(i)?;
        }

        // Initialize gradient
        for (slot, &p_i) in solver.g.iter_mut().zip(&solver.p) {
            *slot = p_i;
        }

        for i in 0..l {
            let status_i = take(&solver.alpha_status, i, "status")?;
            if status_i != AlphaStatus::LowerBound {
                let alpha_i = take(&solver.alpha, i, "alpha")?;
                let q_i = solver.q.get_q(i, l).to_vec();
                let q_i = cut(&q_i, l, "q")?;
                for (slot, &q_ij) in solver.g.iter_mut().zip(q_i.iter()) {
                    *slot += alpha_i * f64::from(q_ij);
                }
                if status_i == AlphaStatus::UpperBound {
                    let c_i = solver.get_c(i)?;
                    for (slot, &q_ij) in solver.g_bar.iter_mut().zip(q_i.iter()) {
                        *slot += c_i * f64::from(q_ij);
                    }
                }
            }
        }

        // Main SMO loop
        // `l` is at most `i32::MAX / 100` in the product branch, so `100 * l` fits.
        let max_iter = 10_000_000usize.max(if l > i32::MAX as usize / 100 {
            usize::MAX
        } else {
            100 * l
        });
        // `l.min(1000)` is at most 1000, so adding one fits in `usize`.
        let mut counter = l.min(1000) + 1;
        let mut iter = 0usize;

        while iter < max_iter {
            // Show progress and do shrinking
            // `counter` is at least one before this subtraction.
            counter -= 1;
            if counter == 0 {
                counter = l.min(1000);
                if shrinking {
                    solver.do_shrinking()?;
                }
            }

            let (wi, wj) = match solver.select_working_set()? {
                Some(pair) => pair,
                None => {
                    // Reconstruct gradient and retry
                    solver.reconstruct_gradient()?;
                    solver.active_size = l;
                    match solver.select_working_set()? {
                        Some(pair) => {
                            counter = 1; // do shrinking next iteration
                            pair
                        }
                        None => break, // optimal
                    }
                }
            };

            // `iter` stays below `max_iter`, so adding one fits in `usize`.
            iter += 1;

            // Update alpha[i] and alpha[j]
            solver.update_alpha_pair(wi, wj)?;
        }

        if iter >= max_iter {
            if solver.active_size < l {
                solver.reconstruct_gradient()?;
                solver.active_size = l;
            }
            crate::info("WARNING: reaching max number of iterations\n");
        }

        // Calculate rho
        let (rho, r) = solver.calculate_rho()?;

        // Calculate objective value
        let obj = {
            let alpha_row = cut(&solver.alpha, l, "alpha")?;
            let grad = cut(&solver.g, l, "g")?;
            let linear = cut(&solver.p, l, "p")?;
            let mut v = 0.0;
            for ((&alpha_i, &g_i), &p_i) in alpha_row.iter().zip(grad).zip(linear) {
                v += alpha_i * (g_i + p_i);
            }
            v / 2.0
        };

        // Put back the solution via active_set mapping
        for i in 0..l {
            let mapped = take(&solver.active_set, i, "active_set")?;
            let alpha_i = take(&solver.alpha, i, "alpha")?;
            *slot(alpha_, mapped, "alpha")? = alpha_i;
        }

        let si = SolutionInfo {
            obj,
            rho,
            upper_bound_p: cp,
            upper_bound_n: cn,
            r,
        };

        crate::info(&format!("optimization finished, #iter = {iter}\n"));

        Ok(si)
    }

    // ─── Helper methods ─────────────────────────────────────────────

    /// Reject a row count above `l`, then require every vector to cover `l`.
    fn rows_active(&self) -> Result<(), RowFault> {
        if self.active_size > self.l {
            return Err(RowFault {
                name: "active_size",
                index: self.active_size,
                len: self.l,
            });
        }
        self.rows(self.l)
    }

    /// Require every solver vector to cover `n`.
    fn rows(&self, n: usize) -> Result<(), RowFault> {
        cut(&self.y, n, "y")?;
        cut(&self.g, n, "g")?;
        cut(&self.g_bar, n, "g_bar")?;
        cut(&self.alpha, n, "alpha")?;
        cut(&self.alpha_status, n, "status")?;
        cut(&self.p, n, "p")?;
        cut(&self.active_set, n, "active_set")?;
        cut(&self.qd, n, "qd")?;
        Ok(())
    }

    #[inline]
    fn get_c(&self, i: usize) -> Result<f64, RowFault> {
        let y_i = take(&self.y, i, "y")?;
        Ok(if y_i > 0 { self.cp } else { self.cn })
    }

    #[inline]
    fn update_alpha_status(&mut self, i: usize) -> Result<(), RowFault> {
        let alpha_i = take(&self.alpha, i, "alpha")?;
        let c_i = self.get_c(i)?;
        let status = if alpha_i >= c_i {
            AlphaStatus::UpperBound
        } else if alpha_i <= 0.0 {
            AlphaStatus::LowerBound
        } else {
            AlphaStatus::Free
        };
        *slot(&mut self.alpha_status, i, "status")? = status;
        Ok(())
    }

    fn swap_index(&mut self, i: usize, j: usize) -> Result<(), RowFault> {
        let index = i.max(j);
        if index
            >= self
                .y
                .len()
                .min(self.g.len())
                .min(self.g_bar.len())
                .min(self.alpha_status.len())
                .min(self.alpha.len())
                .min(self.p.len())
                .min(self.active_set.len())
                .min(self.qd.len())
        {
            let lengths = [
                ("y", self.y.len()),
                ("g", self.g.len()),
                ("g_bar", self.g_bar.len()),
                ("status", self.alpha_status.len()),
                ("alpha", self.alpha.len()),
                ("p", self.p.len()),
                ("active_set", self.active_set.len()),
                ("qd", self.qd.len()),
            ];
            for (name, len) in lengths {
                if index >= len {
                    return Err(RowFault { name, index, len });
                }
            }
        }
        // The combined length check covers both indices for all eight swaps.
        self.q.swap_index(i, j);
        self.y.swap(i, j);
        self.g.swap(i, j);
        self.alpha_status.swap(i, j);
        self.alpha.swap(i, j);
        self.p.swap(i, j);
        self.active_set.swap(i, j);
        self.g_bar.swap(i, j);
        self.qd.swap(i, j);
        Ok(())
    }

    #[expect(
        clippy::arithmetic_side_effects,
        reason = "nr_free <= active_size <= l <= isize::MAX; the comparison products use u128"
    )]
    fn reconstruct_gradient(&mut self) -> Result<(), RowFault> {
        if self.active_size == self.l {
            return Ok(());
        }
        self.rows_active()?;
        let active_size = self.active_size;
        let l = self.l;

        // `j` stays below `l`. `rows_active` checked `g`, `g_bar` and `p`.
        #[expect(
            clippy::indexing_slicing,
            reason = "j stays below l, and rows_active checked g, g_bar and p"
        )]
        for j in active_size..l {
            self.g[j] = self.g_bar[j] + self.p[j];
        }

        let status = cut(&self.alpha_status, active_size, "status")?;
        let mut nr_free: usize = 0;
        for one_status in status {
            if *one_status == AlphaStatus::Free {
                // `nr_free` stays below the active row count, so adding one fits.
                nr_free += 1;
            }
        }

        // `nr_free <= l`, and a slice length is at most `isize::MAX`, so twice the count fits.
        if 2 * nr_free < active_size {
            crate::info("WARNING: using -h 0 may be faster\n");
        }

        let alpha = cut(&self.alpha, active_size, "alpha")?;
        // Both loop orders add each G entry's terms in ascending free-row order, as in C.
        // With l <= isize::MAX, even 2 * l * l fits u128 on 32- and 64-bit targets.
        if (nr_free as u128) * (l as u128) > 2 * (active_size as u128) * ((l - active_size) as u128)
        {
            for i in active_size..l {
                let q_i = self.q.get_q(i, active_size).to_vec();
                let q_i = cut(&q_i, active_size, "q")?;
                #[expect(
                    clippy::indexing_slicing,
                    reason = "rows_active checked g through l and i < l; j < active_size and alpha, status, q_i were cut to it"
                )]
                for j in 0..active_size {
                    if status[j] == AlphaStatus::Free {
                        self.g[i] += alpha[j] * f64::from(q_i[j]);
                    }
                }
            }
        } else {
            for (i, (one_status, &alpha_i)) in status.iter().zip(alpha).enumerate() {
                if *one_status == AlphaStatus::Free {
                    let q_i = self.q.get_q(i, l).to_vec();
                    let q_i = cut(&q_i, l, "q")?;
                    #[expect(
                        clippy::indexing_slicing,
                        reason = "j stays below l, rows_active checked g, and q_i was cut to l"
                    )]
                    for j in active_size..l {
                        self.g[j] += alpha_i * f64::from(q_i[j]);
                    }
                }
            }
        }
        Ok(())
    }

    // ─── Working set selection ──────────────────────────────────────

    /// Return one working-set pair, or none when the set is optimal.
    ///
    /// A short solver vector returns a row fault.
    fn select_working_set(&mut self) -> Result<Option<(usize, usize)>, RowFault> {
        match self.variant {
            SolverVariant::Standard => self.select_working_set_standard(),
            SolverVariant::Nu => self.select_working_set_nu(),
        }
    }

    #[expect(
        clippy::similar_names,
        reason = "Keep the distinct C names for comparison with LIBSVM"
    )]
    #[expect(
        clippy::many_single_char_names,
        reason = "The C solver uses these short names"
    )]
    fn select_working_set_standard(&mut self) -> Result<Option<(usize, usize)>, RowFault> {
        self.rows_active()?;
        let n = self.active_size;
        let y = cut(&self.y, n, "y")?;
        let g = cut(&self.g, n, "g")?;
        let status = cut(&self.alpha_status, n, "status")?;

        let mut gmax = -INF;
        let mut gmax2 = -INF;
        let mut gmax_idx: Option<usize> = None;
        let mut gmin_idx: Option<usize> = None;
        let mut obj_diff_min = INF;

        // First pass: find i (maximizes -y_i * grad(f)_i in I_up)
        #[expect(
            clippy::indexing_slicing,
            reason = "t stays below active_size, and y, g and status were cut to that length"
        )]
        for t in 0..n {
            if y[t] == 1 {
                if status[t] != AlphaStatus::UpperBound && -g[t] >= gmax {
                    gmax = -g[t];
                    gmax_idx = Some(t);
                }
            } else if status[t] != AlphaStatus::LowerBound && g[t] >= gmax {
                gmax = g[t];
                gmax_idx = Some(t);
            }
        }

        let Some(i) = gmax_idx else {
            return Ok(None);
        };
        let q_i_owned = self.q.get_q(i, n).to_vec();
        let q_i = cut(&q_i_owned, n, "q")?;
        let qd = cut(&self.qd, n, "qd")?;
        let qd_i = take(qd, i, "qd")?;
        let yi = f64::from(take(y, i, "y")?);

        // Second pass: find j (minimizes objective decrease)
        #[expect(
            clippy::indexing_slicing,
            reason = "j stays below active_size, and y, g, status, qd and the Q row were cut to that length"
        )]
        for j in 0..n {
            if y[j] == 1 {
                if status[j] != AlphaStatus::LowerBound {
                    let grad_diff = gmax + g[j];
                    if g[j] >= gmax2 {
                        gmax2 = g[j];
                    }
                    if grad_diff > 0.0 {
                        let quad_coef = qd_i + qd[j] - 2.0 * yi * f64::from(q_i[j]);
                        let obj_diff = if quad_coef > 0.0 {
                            -(grad_diff * grad_diff) / quad_coef
                        } else {
                            -(grad_diff * grad_diff) / TAU
                        };
                        if obj_diff <= obj_diff_min {
                            gmin_idx = Some(j);
                            obj_diff_min = obj_diff;
                        }
                    }
                }
            } else if status[j] != AlphaStatus::UpperBound {
                let grad_diff = gmax - g[j];
                if -g[j] >= gmax2 {
                    gmax2 = -g[j];
                }
                if grad_diff > 0.0 {
                    let quad_coef = qd_i + qd[j] + 2.0 * yi * f64::from(q_i[j]);
                    let obj_diff = if quad_coef > 0.0 {
                        -(grad_diff * grad_diff) / quad_coef
                    } else {
                        -(grad_diff * grad_diff) / TAU
                    };
                    if obj_diff <= obj_diff_min {
                        gmin_idx = Some(j);
                        obj_diff_min = obj_diff;
                    }
                }
            }
        }

        if gmax + gmax2 < self.eps {
            return Ok(None);
        }
        let Some(j) = gmin_idx else {
            return Ok(None);
        };
        Ok(Some((i, j)))
    }

    #[expect(
        clippy::similar_names,
        reason = "Keep the distinct C names for comparison with LIBSVM"
    )]
    fn select_working_set_nu(&mut self) -> Result<Option<(usize, usize)>, RowFault> {
        self.rows_active()?;
        let n = self.active_size;
        let y = cut(&self.y, n, "y")?;
        let g = cut(&self.g, n, "g")?;
        let status = cut(&self.alpha_status, n, "status")?;

        let mut gmaxp = -INF;
        let mut gmaxp2 = -INF;
        let mut gmaxp_idx: Option<usize> = None;
        let mut gmaxn = -INF;
        let mut gmaxn2 = -INF;
        let mut gmaxn_idx: Option<usize> = None;
        let mut gmin_idx: Option<usize> = None;
        let mut obj_diff_min = INF;

        #[expect(
            clippy::indexing_slicing,
            reason = "t stays below active_size, and y, g and status were cut to that length"
        )]
        for t in 0..n {
            if y[t] == 1 {
                if status[t] != AlphaStatus::UpperBound && -g[t] >= gmaxp {
                    gmaxp = -g[t];
                    gmaxp_idx = Some(t);
                }
            } else if status[t] != AlphaStatus::LowerBound && g[t] >= gmaxn {
                gmaxn = g[t];
                gmaxn_idx = Some(t);
            }
        }

        let ip = gmaxp_idx;
        let in_ = gmaxn_idx;
        let q_ip_owned = ip.map(|ip| self.q.get_q(ip, n).to_vec());
        let q_in_owned = in_.map(|in_| self.q.get_q(in_, n).to_vec());
        let q_ip = match q_ip_owned.as_ref() {
            Some(row) => Some(cut(row, n, "q")?),
            None => None,
        };
        let q_in = match q_in_owned.as_ref() {
            Some(row) => Some(cut(row, n, "q")?),
            None => None,
        };
        let qd = cut(&self.qd, n, "qd")?;

        #[expect(
            clippy::indexing_slicing,
            reason = "j, ip and in_ come from 0..n; y, g, status, qd and each Q row were cut to n above"
        )]
        for j in 0..n {
            if y[j] == 1 {
                if status[j] != AlphaStatus::LowerBound {
                    let grad_diff = gmaxp + g[j];
                    if g[j] >= gmaxp2 {
                        gmaxp2 = g[j];
                    }
                    if grad_diff > 0.0 {
                        if let (Some(ip), Some(q_ip)) = (ip, q_ip) {
                            let quad_coef = qd[ip] + qd[j] - 2.0 * f64::from(q_ip[j]);
                            let obj_diff = if quad_coef > 0.0 {
                                -(grad_diff * grad_diff) / quad_coef
                            } else {
                                -(grad_diff * grad_diff) / TAU
                            };
                            if obj_diff <= obj_diff_min {
                                gmin_idx = Some(j);
                                obj_diff_min = obj_diff;
                            }
                        }
                    }
                }
            } else if status[j] != AlphaStatus::UpperBound {
                let grad_diff = gmaxn - g[j];
                if -g[j] >= gmaxn2 {
                    gmaxn2 = -g[j];
                }
                if grad_diff > 0.0 {
                    if let (Some(in_), Some(q_in)) = (in_, q_in) {
                        let quad_coef = qd[in_] + qd[j] - 2.0 * f64::from(q_in[j]);
                        let obj_diff = if quad_coef > 0.0 {
                            -(grad_diff * grad_diff) / quad_coef
                        } else {
                            -(grad_diff * grad_diff) / TAU
                        };
                        if obj_diff <= obj_diff_min {
                            gmin_idx = Some(j);
                            obj_diff_min = obj_diff;
                        }
                    }
                }
            }
        }

        if f64::max(gmaxp + gmaxp2, gmaxn + gmaxn2) < self.eps {
            return Ok(None);
        }
        let Some(out_j) = gmin_idx else {
            return Ok(None);
        };
        let y_j = take(y, out_j, "y")?;
        let out_i = if y_j == 1 { gmaxp_idx } else { gmaxn_idx };
        let Some(out_i) = out_i else {
            return Ok(None);
        };
        Ok(Some((out_i, out_j)))
    }

    // ─── Alpha pair update ──────────────────────────────────────────

    #[expect(
        clippy::if_not_else,
        reason = "Keep the C branch order for comparison with the original"
    )]
    #[expect(
        clippy::similar_names,
        reason = "Keep the distinct C names for the two updated rows"
    )]
    #[expect(clippy::too_many_lines, reason = "Keep the C pair update in one place")]
    fn update_alpha_pair(&mut self, i: usize, j: usize) -> Result<(), RowFault> {
        self.rows_active()?;
        let active_size = self.active_size;
        if i >= active_size || j >= active_size {
            return Err(RowFault {
                name: "pair",
                index: i.max(j),
                len: active_size,
            });
        }
        let q_i_owned = self.q.get_q(i, active_size).to_vec();
        let q_j_owned = self.q.get_q(j, active_size).to_vec();
        let q_i = cut(&q_i_owned, active_size, "q")?;
        let q_j = cut(&q_j_owned, active_size, "q")?;
        let c_i = self.get_c(i)?;
        let c_j = self.get_c(j)?;

        let y_i = take(&self.y, i, "y")?;
        let y_j = take(&self.y, j, "y")?;
        let qd_i = take(&self.qd, i, "qd")?;
        let qd_j = take(&self.qd, j, "qd")?;
        let g_i = take(&self.g, i, "g")?;
        let g_j = take(&self.g, j, "g")?;
        let q_ij = take(q_i, j, "q")?;

        let old_alpha_i = take(&self.alpha, i, "alpha")?;
        let old_alpha_j = take(&self.alpha, j, "alpha")?;

        if y_i != y_j {
            let mut quad_coef = qd_i + qd_j + 2.0 * f64::from(q_ij);
            if quad_coef <= 0.0 {
                quad_coef = TAU;
            }
            let delta = (-g_i - g_j) / quad_coef;
            let diff = take(&self.alpha, i, "alpha")? - take(&self.alpha, j, "alpha")?;
            *slot(&mut self.alpha, i, "alpha")? += delta;
            *slot(&mut self.alpha, j, "alpha")? += delta;

            if diff > 0.0 {
                if take(&self.alpha, j, "alpha")? < 0.0 {
                    *slot(&mut self.alpha, j, "alpha")? = 0.0;
                    *slot(&mut self.alpha, i, "alpha")? = diff;
                }
            } else if take(&self.alpha, i, "alpha")? < 0.0 {
                *slot(&mut self.alpha, i, "alpha")? = 0.0;
                *slot(&mut self.alpha, j, "alpha")? = -diff;
            }
            if diff > c_i - c_j {
                if take(&self.alpha, i, "alpha")? > c_i {
                    *slot(&mut self.alpha, i, "alpha")? = c_i;
                    *slot(&mut self.alpha, j, "alpha")? = c_i - diff;
                }
            } else if take(&self.alpha, j, "alpha")? > c_j {
                *slot(&mut self.alpha, j, "alpha")? = c_j;
                *slot(&mut self.alpha, i, "alpha")? = c_j + diff;
            }
        } else {
            let mut quad_coef = qd_i + qd_j - 2.0 * f64::from(q_ij);
            if quad_coef <= 0.0 {
                quad_coef = TAU;
            }
            let delta = (g_i - g_j) / quad_coef;
            let sum = take(&self.alpha, i, "alpha")? + take(&self.alpha, j, "alpha")?;
            *slot(&mut self.alpha, i, "alpha")? -= delta;
            *slot(&mut self.alpha, j, "alpha")? += delta;

            if sum > c_i {
                if take(&self.alpha, i, "alpha")? > c_i {
                    *slot(&mut self.alpha, i, "alpha")? = c_i;
                    *slot(&mut self.alpha, j, "alpha")? = sum - c_i;
                }
            } else if take(&self.alpha, j, "alpha")? < 0.0 {
                *slot(&mut self.alpha, j, "alpha")? = 0.0;
                *slot(&mut self.alpha, i, "alpha")? = sum;
            }
            if sum > c_j {
                if take(&self.alpha, j, "alpha")? > c_j {
                    *slot(&mut self.alpha, j, "alpha")? = c_j;
                    *slot(&mut self.alpha, i, "alpha")? = sum - c_j;
                }
            } else if take(&self.alpha, i, "alpha")? < 0.0 {
                *slot(&mut self.alpha, i, "alpha")? = 0.0;
                *slot(&mut self.alpha, j, "alpha")? = sum;
            }
        }

        let delta_alpha_i = take(&self.alpha, i, "alpha")? - old_alpha_i;
        let delta_alpha_j = take(&self.alpha, j, "alpha")? - old_alpha_j;
        for ((g_k, &q_ik), &q_jk) in self.g.iter_mut().zip(q_i).zip(q_j) {
            *g_k += f64::from(q_ik) * delta_alpha_i + f64::from(q_jk) * delta_alpha_j;
        }

        let ui = take(&self.alpha_status, i, "status")? == AlphaStatus::UpperBound;
        let uj = take(&self.alpha_status, j, "status")? == AlphaStatus::UpperBound;
        self.update_alpha_status(i)?;
        self.update_alpha_status(j)?;
        let ui_now = take(&self.alpha_status, i, "status")? == AlphaStatus::UpperBound;
        let uj_now = take(&self.alpha_status, j, "status")? == AlphaStatus::UpperBound;
        let l = self.l;

        if ui != ui_now {
            let q_i_full = self.q.get_q(i, l).to_vec();
            let q_i_full = cut(&q_i_full, l, "q")?;
            if ui {
                for (g_bar_k, &q_k) in self.g_bar.iter_mut().zip(q_i_full) {
                    *g_bar_k -= c_i * f64::from(q_k);
                }
            } else {
                for (g_bar_k, &q_k) in self.g_bar.iter_mut().zip(q_i_full) {
                    *g_bar_k += c_i * f64::from(q_k);
                }
            }
        }

        if uj != uj_now {
            let q_j_full = self.q.get_q(j, l).to_vec();
            let q_j_full = cut(&q_j_full, l, "q")?;
            if uj {
                for (g_bar_k, &q_k) in self.g_bar.iter_mut().zip(q_j_full) {
                    *g_bar_k -= c_j * f64::from(q_k);
                }
            } else {
                for (g_bar_k, &q_k) in self.g_bar.iter_mut().zip(q_j_full) {
                    *g_bar_k += c_j * f64::from(q_k);
                }
            }
        }
        Ok(())
    }

    // ─── Shrinking ──────────────────────────────────────────────────

    fn do_shrinking(&mut self) -> Result<(), RowFault> {
        match self.variant {
            SolverVariant::Standard => self.do_shrinking_standard(),
            SolverVariant::Nu => self.do_shrinking_nu(),
        }
    }

    /// Tell whether one row leaves the active set.
    fn be_shrunk_standard(y_i: i8, g_i: f64, status: AlphaStatus, gmax1: f64, gmax2: f64) -> bool {
        if status == AlphaStatus::UpperBound {
            if y_i == 1 { -g_i > gmax1 } else { -g_i > gmax2 }
        } else if status == AlphaStatus::LowerBound {
            if y_i == 1 { g_i > gmax2 } else { g_i > gmax1 }
        } else {
            false
        }
    }

    #[expect(
        clippy::arithmetic_side_effects,
        reason = "active_size is positive before subtraction; i is below the old active_size <= l before increment"
    )]
    fn do_shrinking_standard(&mut self) -> Result<(), RowFault> {
        self.rows_active()?;
        let n = self.active_size;
        let y = cut(&self.y, n, "y")?;
        let g = cut(&self.g, n, "g")?;
        let status = cut(&self.alpha_status, n, "status")?;
        let mut gmax1 = -INF;
        let mut gmax2 = -INF;

        #[expect(
            clippy::indexing_slicing,
            reason = "i stays below active_size, and y, g and status were cut to that length"
        )]
        for i in 0..n {
            if y[i] == 1 {
                if status[i] != AlphaStatus::UpperBound && -g[i] >= gmax1 {
                    gmax1 = -g[i];
                }
                if status[i] != AlphaStatus::LowerBound && g[i] >= gmax2 {
                    gmax2 = g[i];
                }
            } else {
                if status[i] != AlphaStatus::UpperBound && -g[i] >= gmax2 {
                    gmax2 = -g[i];
                }
                if status[i] != AlphaStatus::LowerBound && g[i] >= gmax1 {
                    gmax1 = g[i];
                }
            }
        }

        if !self.unshrink && gmax1 + gmax2 <= self.eps * 10.0 {
            self.unshrink = true;
            self.reconstruct_gradient()?;
            self.active_size = self.l;
        }

        let mut i = 0;
        while i < self.active_size {
            #[expect(
                clippy::indexing_slicing,
                reason = "rows_active checked y, g and status through l; i < active_size <= l"
            )]
            let (y_i, g_i, status_i) = (self.y[i], self.g[i], self.alpha_status[i]);
            if Self::be_shrunk_standard(y_i, g_i, status_i, gmax1, gmax2) {
                // i < active_size, so the active count is positive before subtraction.
                self.active_size -= 1;
                while self.active_size > i {
                    let tail = self.active_size;
                    #[expect(
                        clippy::indexing_slicing,
                        reason = "rows_active checked y, g and status through l; tail is below the old active_size <= l"
                    )]
                    let (y_t, g_t, status_t) =
                        (self.y[tail], self.g[tail], self.alpha_status[tail]);
                    if !Self::be_shrunk_standard(y_t, g_t, status_t, gmax1, gmax2) {
                        self.swap_index(i, tail)?;
                        break;
                    }
                    // active_size > i, so the active count is positive before subtraction.
                    self.active_size -= 1;
                }
            }
            // i < the old active_size <= l <= isize::MAX, so adding one fits.
            i += 1;
        }
        Ok(())
    }

    /// Tell whether one nu row leaves the active set.
    fn be_shrunk_nu(
        y_i: i8,
        g_i: f64,
        status: AlphaStatus,
        gmax1: f64,
        gmax2: f64,
        gmax3: f64,
        gmax4: f64,
    ) -> bool {
        if status == AlphaStatus::UpperBound {
            if y_i == 1 { -g_i > gmax1 } else { -g_i > gmax4 }
        } else if status == AlphaStatus::LowerBound {
            if y_i == 1 { g_i > gmax2 } else { g_i > gmax3 }
        } else {
            false
        }
    }

    #[expect(
        clippy::arithmetic_side_effects,
        reason = "active_size is positive before subtraction; i is below the old active_size <= l before increment"
    )]
    fn do_shrinking_nu(&mut self) -> Result<(), RowFault> {
        self.rows_active()?;
        let n = self.active_size;
        let y = cut(&self.y, n, "y")?;
        let g = cut(&self.g, n, "g")?;
        let status = cut(&self.alpha_status, n, "status")?;
        let mut gmax1 = -INF;
        let mut gmax2 = -INF;
        let mut gmax3 = -INF;
        let mut gmax4 = -INF;

        #[expect(
            clippy::indexing_slicing,
            reason = "i stays below active_size, and y, g and status were cut to that length"
        )]
        for i in 0..n {
            if status[i] != AlphaStatus::UpperBound {
                if y[i] == 1 {
                    if -g[i] > gmax1 {
                        gmax1 = -g[i];
                    }
                } else if -g[i] > gmax4 {
                    gmax4 = -g[i];
                }
            }
            if status[i] != AlphaStatus::LowerBound {
                if y[i] == 1 {
                    if g[i] > gmax2 {
                        gmax2 = g[i];
                    }
                } else if g[i] > gmax3 {
                    gmax3 = g[i];
                }
            }
        }

        if !self.unshrink && f64::max(gmax1 + gmax2, gmax3 + gmax4) <= self.eps * 10.0 {
            self.unshrink = true;
            self.reconstruct_gradient()?;
            self.active_size = self.l;
        }

        let mut i = 0;
        while i < self.active_size {
            #[expect(
                clippy::indexing_slicing,
                reason = "rows_active checked y, g and status through l; i < active_size <= l"
            )]
            let (y_i, g_i, status_i) = (self.y[i], self.g[i], self.alpha_status[i]);
            if Self::be_shrunk_nu(y_i, g_i, status_i, gmax1, gmax2, gmax3, gmax4) {
                // i < active_size, so the active count is positive before subtraction.
                self.active_size -= 1;
                while self.active_size > i {
                    let tail = self.active_size;
                    #[expect(
                        clippy::indexing_slicing,
                        reason = "rows_active checked y, g and status through l; tail is below the old active_size <= l"
                    )]
                    let (y_t, g_t, status_t) =
                        (self.y[tail], self.g[tail], self.alpha_status[tail]);
                    if !Self::be_shrunk_nu(y_t, g_t, status_t, gmax1, gmax2, gmax3, gmax4) {
                        self.swap_index(i, tail)?;
                        break;
                    }
                    // active_size > i, so the active count is positive before subtraction.
                    self.active_size -= 1;
                }
            }
            // i < the old active_size <= l <= isize::MAX, so adding one fits.
            i += 1;
        }
        Ok(())
    }

    // ─── Rho calculation ────────────────────────────────────────────

    fn calculate_rho(&self) -> Result<(f64, f64), RowFault> {
        match self.variant {
            SolverVariant::Standard => Ok((self.calculate_rho_standard()?, 0.0)),
            SolverVariant::Nu => self.calculate_rho_nu(),
        }
    }

    #[expect(
        clippy::manual_midpoint,
        reason = "Keep the C midpoint arithmetic and operation order"
    )]
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "each free-row count is at most active_size <= l <= isize::MAX"
    )]
    fn calculate_rho_standard(&self) -> Result<f64, RowFault> {
        self.rows_active()?;
        let n = self.active_size;
        let y = cut(&self.y, n, "y")?;
        let g = cut(&self.g, n, "g")?;
        let status = cut(&self.alpha_status, n, "status")?;
        let mut nr_free: usize = 0;
        let mut ub = INF;
        let mut lb = -INF;
        let mut sum_free = 0.0;

        #[expect(
            clippy::indexing_slicing,
            reason = "i stays below active_size, and y, g and status were cut to that length"
        )]
        for i in 0..n {
            let yg = f64::from(y[i]) * g[i];

            if status[i] == AlphaStatus::UpperBound {
                if y[i] == -1 {
                    ub = ub.min(yg);
                } else {
                    lb = lb.max(yg);
                }
            } else if status[i] == AlphaStatus::LowerBound {
                if y[i] == 1 {
                    ub = ub.min(yg);
                } else {
                    lb = lb.max(yg);
                }
            } else {
                // The count is at most active_size, so adding one fits.
                nr_free += 1;
                sum_free += yg;
            }
        }

        if nr_free > 0 {
            #[expect(
                clippy::cast_precision_loss,
                reason = "the row count is converted once at division; run rejects more than 2^53 rows, so the count is exact"
            )]
            let count = nr_free as f64;
            Ok(sum_free / count)
        } else {
            Ok((ub + lb) / 2.0)
        }
    }

    #[expect(
        clippy::manual_midpoint,
        reason = "Keep the C midpoint arithmetic and operation order"
    )]
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "each free-row count is at most active_size <= l <= isize::MAX"
    )]
    fn calculate_rho_nu(&self) -> Result<(f64, f64), RowFault> {
        self.rows_active()?;
        let n = self.active_size;
        let y = cut(&self.y, n, "y")?;
        let g = cut(&self.g, n, "g")?;
        let status = cut(&self.alpha_status, n, "status")?;
        let mut nr_free1: usize = 0;
        let mut nr_free2: usize = 0;
        let mut ub1 = INF;
        let mut ub2 = INF;
        let mut lb1 = -INF;
        let mut lb2 = -INF;
        let mut sum_free1 = 0.0;
        let mut sum_free2 = 0.0;

        #[expect(
            clippy::indexing_slicing,
            reason = "i stays below active_size, and y, g and status were cut to that length"
        )]
        for i in 0..n {
            if y[i] == 1 {
                if status[i] == AlphaStatus::UpperBound {
                    lb1 = lb1.max(g[i]);
                } else if status[i] == AlphaStatus::LowerBound {
                    ub1 = ub1.min(g[i]);
                } else {
                    // The count is at most active_size, so adding one fits.
                    nr_free1 += 1;
                    sum_free1 += g[i];
                }
            } else if status[i] == AlphaStatus::UpperBound {
                lb2 = lb2.max(g[i]);
            } else if status[i] == AlphaStatus::LowerBound {
                ub2 = ub2.min(g[i]);
            } else {
                // The count is at most active_size, so adding one fits.
                nr_free2 += 1;
                sum_free2 += g[i];
            }
        }

        let r1 = if nr_free1 > 0 {
            #[expect(
                clippy::cast_precision_loss,
                reason = "the row count is converted once at division; run rejects more than 2^53 rows, so the count is exact"
            )]
            let count = nr_free1 as f64;
            sum_free1 / count
        } else {
            (ub1 + lb1) / 2.0
        };

        let r2 = if nr_free2 > 0 {
            #[expect(
                clippy::cast_precision_loss,
                reason = "the row count is converted once at division; run rejects more than 2^53 rows, so the count is exact"
            )]
            let count = nr_free2 as f64;
            sum_free2 / count
        } else {
            (ub2 + lb2) / 2.0
        };

        let rho = (r1 - r2) / 2.0;
        let r = (r1 + r2) / 2.0;
        Ok((rho, r))
    }
}

#[cfg(test)]
mod tests {
    //! Check that a short solver vector returns a row fault.

    use super::{AlphaStatus, RowFault, Solver, SolverVariant, cut, slot, take};
    use crate::cache::Qfloat;
    use crate::qmatrix::QMatrix;

    struct StubQ {
        qd: Vec<f64>,
        row: Vec<Qfloat>,
    }

    impl QMatrix for StubQ {
        fn get_q(&mut self, _i: usize, len: usize) -> &[Qfloat] {
            if self.row.len() < len {
                self.row.resize(len, 0.0);
            }
            self.row.get(..len).unwrap_or(&[])
        }

        fn get_qd(&self) -> &[f64] {
            &self.qd
        }

        fn swap_index(&mut self, i: usize, j: usize) {
            if i < self.qd.len() && j < self.qd.len() {
                self.qd.swap(i, j);
            }
            if i < self.row.len() && j < self.row.len() {
                self.row.swap(i, j);
            }
        }
    }

    fn bare(variant: SolverVariant, row_count: usize, label_count: usize) -> Solver<'static> {
        let qd = vec![1.0; row_count];
        Solver {
            l: row_count,
            active_size: row_count,
            variant,
            y: vec![1_i8; label_count],
            g: vec![0.0; row_count],
            g_bar: vec![0.0; row_count],
            alpha: vec![0.0; row_count],
            alpha_status: vec![AlphaStatus::LowerBound; row_count],
            p: vec![0.0; row_count],
            active_set: (0..row_count).collect(),
            unshrink: false,
            q: Box::new(StubQ {
                qd: qd.clone(),
                row: vec![0.0; row_count],
            }),
            qd,
            cp: 1.0,
            cn: 1.0,
            eps: 0.001,
        }
    }

    fn short_labels(variant: SolverVariant) -> Solver<'static> {
        bare(variant, 4, 1)
    }

    fn assert_fault<T>(result: Result<T, RowFault>, name: &str, index: usize, len: usize) {
        assert!(result.is_err());
        let Err(fault) = result else {
            return;
        };
        assert_eq!(fault.name, name);
        assert_eq!(fault.index, index);
        assert_eq!(fault.len, len);
    }

    #[test]
    fn cut_rejects_a_short_prefix() {
        let row = [0.0, 1.0];
        assert_fault(cut(&row[..], 4, "g").map(|_| ()), "g", 4, 2);
    }

    #[test]
    fn take_rejects_a_missing_index() {
        let row = [0.0, 1.0];
        assert_fault(take(&row[..], 3, "g"), "g", 3, 2);
    }

    #[test]
    fn slot_rejects_a_missing_index() {
        let mut row = [0.0, 1.0];
        assert_fault(slot(&mut row[..], 3, "alpha").map(|_| ()), "alpha", 3, 2);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    #[should_panic(expected = "above 2^53")]
    fn solve_panics_with_a_clear_message_above_two_to_the_53_rows() {
        let q: Box<dyn QMatrix> = Box::new(StubQ {
            qd: vec![1.0; 2],
            row: vec![0.0; 2],
        });
        let mut alpha = vec![0.0; 2];
        let _ = Solver::solve(
            SolverVariant::Standard,
            usize::MAX,
            q,
            &[0.0; 2],
            &[1_i8; 2],
            &mut alpha,
            1.0,
            1.0,
            0.001,
            true,
        );
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn run_rejects_more_than_two_to_the_53_rows() {
        let q: Box<dyn QMatrix> = Box::new(StubQ {
            qd: vec![1.0; 2],
            row: vec![0.0; 2],
        });
        let mut alpha = vec![0.0; 2];
        let result = Solver::run(
            SolverVariant::Standard,
            usize::MAX,
            q,
            &[0.0; 2],
            &[1_i8; 2],
            &mut alpha,
            1.0,
            1.0,
            0.001,
            true,
        );
        assert_fault(result.map(|_| ()), "l", usize::MAX, usize::MAX);
    }

    #[test]
    fn rows_rejects_a_short_label_row() {
        let solver = short_labels(SolverVariant::Standard);
        assert_fault(solver.rows(4), "y", 4, 1);
    }

    #[test]
    fn rows_active_rejects_a_short_label_row() {
        let solver = short_labels(SolverVariant::Standard);
        assert_fault(solver.rows_active(), "y", 4, 1);
    }

    #[test]
    fn rows_active_rejects_a_count_above_the_row_count() {
        let mut solver = bare(SolverVariant::Standard, 4, 4);
        solver.active_size = 9;
        assert_fault(solver.rows_active(), "active_size", 9, 4);
    }

    #[test]
    fn get_c_rejects_an_empty_label_row() {
        let solver = bare(SolverVariant::Standard, 0, 0);
        assert_fault(solver.get_c(0), "y", 0, 0);
    }

    #[test]
    fn update_alpha_status_rejects_an_empty_alpha_row() {
        let mut solver = bare(SolverVariant::Standard, 4, 4);
        solver.alpha.clear();
        assert_fault(solver.update_alpha_status(0), "alpha", 0, 0);
    }

    #[test]
    fn swap_index_rejects_a_missing_column() {
        let mut solver = bare(SolverVariant::Standard, 4, 4);
        assert_fault(solver.swap_index(0, 9), "y", 9, 4);
        assert_eq!(solver.y.first().copied(), Some(1));
    }

    #[test]
    fn swap_index_exchanges_two_labels() {
        let mut solver = bare(SolverVariant::Standard, 4, 4);
        solver.y = vec![1, -1, 1, 1];
        assert!(solver.swap_index(0, 1).is_ok());
        assert_eq!(solver.y.first().copied(), Some(-1));
        assert_eq!(solver.y.get(1).copied(), Some(1));
    }

    #[test]
    fn reconstruct_gradient_rejects_a_short_label_row() {
        let mut solver = short_labels(SolverVariant::Standard);
        solver.active_size = 3;
        assert_fault(solver.reconstruct_gradient(), "y", 4, 1);
    }

    #[test]
    fn select_working_set_rejects_a_short_label_row() {
        let mut solver = short_labels(SolverVariant::Standard);
        assert_fault(solver.select_working_set(), "y", 4, 1);
    }

    #[test]
    fn select_working_set_nu_variant_rejects_a_short_label_row() {
        let mut solver = short_labels(SolverVariant::Nu);
        assert_fault(solver.select_working_set(), "y", 4, 1);
    }

    #[test]
    fn select_working_set_standard_rejects_a_short_label_row() {
        let mut solver = short_labels(SolverVariant::Standard);
        assert_fault(solver.select_working_set_standard(), "y", 4, 1);
    }

    #[test]
    fn select_working_set_nu_rejects_a_short_label_row() {
        let mut solver = short_labels(SolverVariant::Nu);
        assert_fault(solver.select_working_set_nu(), "y", 4, 1);
    }

    #[test]
    fn select_working_set_standard_rejects_an_active_count_above_the_row_count() {
        let mut solver = bare(SolverVariant::Standard, 4, 4);
        solver.active_size = 9;
        assert_fault(solver.select_working_set_standard(), "active_size", 9, 4);
    }

    #[test]
    fn update_alpha_pair_rejects_a_short_label_row() {
        let mut solver = short_labels(SolverVariant::Standard);
        assert_fault(solver.update_alpha_pair(0, 0), "y", 4, 1);
    }

    #[test]
    fn update_alpha_pair_rejects_an_index_past_the_active_count() {
        let mut solver = bare(SolverVariant::Standard, 4, 4);
        solver.active_size = 2;
        assert_fault(solver.update_alpha_pair(0, 3), "pair", 3, 2);
    }

    #[test]
    fn do_shrinking_rejects_a_short_label_row() {
        let mut solver = short_labels(SolverVariant::Standard);
        assert_fault(solver.do_shrinking(), "y", 4, 1);
    }

    #[test]
    fn do_shrinking_nu_variant_rejects_a_short_label_row() {
        let mut solver = short_labels(SolverVariant::Nu);
        assert_fault(solver.do_shrinking(), "y", 4, 1);
    }

    #[test]
    fn do_shrinking_standard_rejects_a_short_label_row() {
        let mut solver = short_labels(SolverVariant::Standard);
        assert_fault(solver.do_shrinking_standard(), "y", 4, 1);
    }

    #[test]
    fn do_shrinking_nu_rejects_a_short_label_row() {
        let mut solver = short_labels(SolverVariant::Nu);
        assert_fault(solver.do_shrinking_nu(), "y", 4, 1);
    }

    #[test]
    fn calculate_rho_rejects_a_short_label_row() {
        let solver = short_labels(SolverVariant::Standard);
        assert_fault(solver.calculate_rho(), "y", 4, 1);
    }

    #[test]
    fn calculate_rho_nu_variant_rejects_a_short_label_row() {
        let solver = short_labels(SolverVariant::Nu);
        assert_fault(solver.calculate_rho(), "y", 4, 1);
    }

    #[test]
    fn calculate_rho_standard_rejects_a_short_label_row() {
        let solver = short_labels(SolverVariant::Standard);
        assert_fault(solver.calculate_rho_standard(), "y", 4, 1);
    }

    #[test]
    fn calculate_rho_nu_rejects_a_short_label_row() {
        let solver = short_labels(SolverVariant::Nu);
        assert_fault(solver.calculate_rho_nu(), "y", 4, 1);
    }

    #[test]
    fn run_rejects_a_short_linear_row() {
        let q = Box::new(StubQ {
            qd: vec![1.0, 1.0],
            row: vec![0.0, 0.0],
        });
        let y = [1_i8, -1];
        let p: [f64; 0] = [];
        let mut alpha = [0.0, 0.0];
        let result = Solver::run(
            SolverVariant::Standard,
            2,
            q,
            &p,
            &y,
            &mut alpha,
            1.0,
            1.0,
            0.001,
            false,
        );
        assert_fault(result, "p", 2, 0);
    }
}
