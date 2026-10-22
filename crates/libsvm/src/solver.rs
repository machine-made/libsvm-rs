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

#[allow(clippy::needless_range_loop)]
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
    #[allow(clippy::too_many_arguments)]
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
        let qd = q.get_qd().to_vec();
        let p = p_.to_vec();
        let y = y_.to_vec();
        let alpha = alpha_.to_vec();

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
            solver.update_alpha_status(i);
        }

        // Initialize gradient
        for i in 0..l {
            solver.g[i] = solver.p[i];
        }

        for i in 0..l {
            if !solver.is_lower_bound(i) {
                let alpha_i = solver.alpha[i];
                let q_i = solver.q.get_q(i, l).to_vec();
                for j in 0..l {
                    solver.g[j] += alpha_i * q_i[j] as f64;
                }
                if solver.is_upper_bound(i) {
                    let c_i = solver.get_c(i);
                    for j in 0..l {
                        solver.g_bar[j] += c_i * q_i[j] as f64;
                    }
                }
            }
        }

        // Main SMO loop
        let max_iter = 10_000_000usize.max(if l > i32::MAX as usize / 100 {
            usize::MAX
        } else {
            100 * l
        });
        let mut counter = l.min(1000) + 1;
        let mut iter = 0usize;

        while iter < max_iter {
            // Show progress and do shrinking
            counter -= 1;
            if counter == 0 {
                counter = l.min(1000);
                if shrinking {
                    solver.do_shrinking();
                }
            }

            let (wi, wj) = match solver.select_working_set() {
                Some(pair) => pair,
                None => {
                    // Reconstruct gradient and retry
                    solver.reconstruct_gradient();
                    solver.active_size = l;
                    match solver.select_working_set() {
                        Some(pair) => {
                            counter = 1; // do shrinking next iteration
                            pair
                        }
                        None => break, // optimal
                    }
                }
            };

            iter += 1;

            // Update alpha[i] and alpha[j]
            solver.update_alpha_pair(wi, wj);
        }

        if iter >= max_iter {
            if solver.active_size < l {
                solver.reconstruct_gradient();
                solver.active_size = l;
            }
            crate::info("WARNING: reaching max number of iterations\n");
        }

        // Calculate rho
        let (rho, r) = solver.calculate_rho();

        // Calculate objective value
        let obj = {
            let mut v = 0.0;
            for i in 0..l {
                v += solver.alpha[i] * (solver.g[i] + solver.p[i]);
            }
            v / 2.0
        };

        // Put back the solution via active_set mapping
        for i in 0..l {
            alpha_[solver.active_set[i]] = solver.alpha[i];
        }

        let si = SolutionInfo {
            obj,
            rho,
            upper_bound_p: cp,
            upper_bound_n: cn,
            r,
        };

        crate::info(&format!("optimization finished, #iter = {}\n", iter));

        si
    }

    // ─── Helper methods ─────────────────────────────────────────────

    #[inline]
    fn get_c(&self, i: usize) -> f64 {
        if self.y[i] > 0 {
            self.cp
        } else {
            self.cn
        }
    }

    #[inline]
    fn update_alpha_status(&mut self, i: usize) {
        self.alpha_status[i] = if self.alpha[i] >= self.get_c(i) {
            AlphaStatus::UpperBound
        } else if self.alpha[i] <= 0.0 {
            AlphaStatus::LowerBound
        } else {
            AlphaStatus::Free
        };
    }

    #[inline]
    fn is_upper_bound(&self, i: usize) -> bool {
        self.alpha_status[i] == AlphaStatus::UpperBound
    }

    #[inline]
    fn is_lower_bound(&self, i: usize) -> bool {
        self.alpha_status[i] == AlphaStatus::LowerBound
    }

    #[inline]
    fn is_free(&self, i: usize) -> bool {
        self.alpha_status[i] == AlphaStatus::Free
    }

    fn swap_index(&mut self, i: usize, j: usize) {
        self.q.swap_index(i, j);
        self.y.swap(i, j);
        self.g.swap(i, j);
        self.alpha_status.swap(i, j);
        self.alpha.swap(i, j);
        self.p.swap(i, j);
        self.active_set.swap(i, j);
        self.g_bar.swap(i, j);
        self.qd.swap(i, j);
    }

    fn reconstruct_gradient(&mut self) {
        if self.active_size == self.l {
            return;
        }

        for j in self.active_size..self.l {
            self.g[j] = self.g_bar[j] + self.p[j];
        }

        let mut nr_free = 0;
        for j in 0..self.active_size {
            if self.is_free(j) {
                nr_free += 1;
            }
        }

        if 2 * nr_free < self.active_size {
            crate::info("WARNING: using -h 0 may be faster\n");
        }

        let active_size = self.active_size;
        let l = self.l;

        if nr_free * l > 2 * active_size * (l - active_size) {
            for i in active_size..l {
                let q_i = self.q.get_q(i, active_size).to_vec();
                for j in 0..active_size {
                    if self.is_free(j) {
                        self.g[i] += self.alpha[j] * q_i[j] as f64;
                    }
                }
            }
        } else {
            for i in 0..active_size {
                if self.is_free(i) {
                    let q_i = self.q.get_q(i, l).to_vec();
                    let alpha_i = self.alpha[i];
                    for j in active_size..l {
                        self.g[j] += alpha_i * q_i[j] as f64;
                    }
                }
            }
        }
    }

    // ─── Working set selection ──────────────────────────────────────

    /// Select working set (i, j). Returns None if already optimal.
    fn select_working_set(&mut self) -> Option<(usize, usize)> {
        match self.variant {
            SolverVariant::Standard => self.select_working_set_standard(),
            SolverVariant::Nu => self.select_working_set_nu(),
        }
    }

    fn select_working_set_standard(&mut self) -> Option<(usize, usize)> {
        let mut gmax = -INF;
        let mut gmax2 = -INF;
        let mut gmax_idx: Option<usize> = None;
        let mut gmin_idx: Option<usize> = None;
        let mut obj_diff_min = INF;

        // First pass: find i (maximizes -y_i * grad(f)_i in I_up)
        // Slice to active_size so the compiler can drop per-index bounds checks.
        let n = self.active_size;
        let (y, g, status) = (&self.y[..n], &self.g[..n], &self.alpha_status[..n]);
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

        let i = gmax_idx?;
        let q_i = self.q.get_q(i, self.active_size).to_vec();

        // Second pass: find j (minimizes objective decrease)
        // Reuse `y`/`g`/`status` sliced to `n`; index values are < n.
        let qd_i = self.qd[i];
        let yi = y[i] as f64;
        for j in 0..n {
            if y[j] == 1 {
                if status[j] != AlphaStatus::LowerBound {
                    let grad_diff = gmax + g[j];
                    if g[j] >= gmax2 {
                        gmax2 = g[j];
                    }
                    if grad_diff > 0.0 {
                        let quad_coef = qd_i + self.qd[j] - 2.0 * yi * q_i[j] as f64;
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
                    let quad_coef = qd_i + self.qd[j] + 2.0 * yi * q_i[j] as f64;
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

        if gmax + gmax2 < self.eps || gmin_idx.is_none() {
            return None;
        }

        Some((i, gmin_idx?))
    }

    fn select_working_set_nu(&mut self) -> Option<(usize, usize)> {
        let mut gmaxp = -INF;
        let mut gmaxp2 = -INF;
        let mut gmaxp_idx: Option<usize> = None;
        let mut gmaxn = -INF;
        let mut gmaxn2 = -INF;
        let mut gmaxn_idx: Option<usize> = None;
        let mut gmin_idx: Option<usize> = None;
        let mut obj_diff_min = INF;

        // Slice to active_size so index-based loops drop bounds checks.
        let n = self.active_size;
        let (y, g, status) = (&self.y[..n], &self.g[..n], &self.alpha_status[..n]);
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

        let q_ip = if let Some(ip) = ip {
            Some(self.q.get_q(ip, self.active_size).to_vec())
        } else {
            None
        };
        let q_in = if let Some(in_) = in_ {
            Some(self.q.get_q(in_, self.active_size).to_vec())
        } else {
            None
        };

        for j in 0..n {
            if y[j] == 1 {
                if status[j] != AlphaStatus::LowerBound {
                    let grad_diff = gmaxp + g[j];
                    if g[j] >= gmaxp2 {
                        gmaxp2 = g[j];
                    }
                    if grad_diff > 0.0 {
                        if let (Some(ip), Some(q_ip)) = (ip, q_ip.as_ref()) {
                            let quad_coef = self.qd[ip] + self.qd[j] - 2.0 * q_ip[j] as f64;
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
                    if let (Some(in_), Some(q_in)) = (in_, q_in.as_ref()) {
                        let quad_coef = self.qd[in_] + self.qd[j] - 2.0 * q_in[j] as f64;
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

        if f64::max(gmaxp + gmaxp2, gmaxn + gmaxn2) < self.eps || gmin_idx.is_none() {
            return None;
        }

        let out_j = gmin_idx?;
        let out_i = if self.y[out_j] == 1 {
            gmaxp_idx?
        } else {
            gmaxn_idx?
        };

        Some((out_i, out_j))
    }

    // ─── Alpha pair update ──────────────────────────────────────────

    fn update_alpha_pair(&mut self, i: usize, j: usize) {
        let active_size = self.active_size;
        let q_i = self.q.get_q(i, active_size).to_vec();
        let q_j = self.q.get_q(j, active_size).to_vec();

        let c_i = self.get_c(i);
        let c_j = self.get_c(j);

        let old_alpha_i = self.alpha[i];
        let old_alpha_j = self.alpha[j];

        if self.y[i] != self.y[j] {
            let mut quad_coef = self.qd[i] + self.qd[j] + 2.0 * q_i[j] as f64;
            if quad_coef <= 0.0 {
                quad_coef = TAU;
            }
            let delta = (-self.g[i] - self.g[j]) / quad_coef;
            let diff = self.alpha[i] - self.alpha[j];
            self.alpha[i] += delta;
            self.alpha[j] += delta;

            if diff > 0.0 {
                if self.alpha[j] < 0.0 {
                    self.alpha[j] = 0.0;
                    self.alpha[i] = diff;
                }
            } else if self.alpha[i] < 0.0 {
                self.alpha[i] = 0.0;
                self.alpha[j] = -diff;
            }
            if diff > c_i - c_j {
                if self.alpha[i] > c_i {
                    self.alpha[i] = c_i;
                    self.alpha[j] = c_i - diff;
                }
            } else if self.alpha[j] > c_j {
                self.alpha[j] = c_j;
                self.alpha[i] = c_j + diff;
            }
        } else {
            let mut quad_coef = self.qd[i] + self.qd[j] - 2.0 * q_i[j] as f64;
            if quad_coef <= 0.0 {
                quad_coef = TAU;
            }
            let delta = (self.g[i] - self.g[j]) / quad_coef;
            let sum = self.alpha[i] + self.alpha[j];
            self.alpha[i] -= delta;
            self.alpha[j] += delta;

            if sum > c_i {
                if self.alpha[i] > c_i {
                    self.alpha[i] = c_i;
                    self.alpha[j] = sum - c_i;
                }
            } else if self.alpha[j] < 0.0 {
                self.alpha[j] = 0.0;
                self.alpha[i] = sum;
            }
            if sum > c_j {
                if self.alpha[j] > c_j {
                    self.alpha[j] = c_j;
                    self.alpha[i] = sum - c_j;
                }
            } else if self.alpha[i] < 0.0 {
                self.alpha[i] = 0.0;
                self.alpha[j] = sum;
            }
        }

        // Update gradient G
        let delta_alpha_i = self.alpha[i] - old_alpha_i;
        let delta_alpha_j = self.alpha[j] - old_alpha_j;

        // Update gradient G (sliced so `k < active_size` proves in-bounds)
        let g = &mut self.g[..active_size];
        for k in 0..active_size {
            g[k] += q_i[k] as f64 * delta_alpha_i + q_j[k] as f64 * delta_alpha_j;
        }

        // Update alpha_status and G_bar
        let ui = self.is_upper_bound(i);
        let uj = self.is_upper_bound(j);
        self.update_alpha_status(i);
        self.update_alpha_status(j);

        let l = self.l;

        if ui != self.is_upper_bound(i) {
            let q_i_full = self.q.get_q(i, l).to_vec();
            if ui {
                for k in 0..l {
                    self.g_bar[k] -= c_i * q_i_full[k] as f64;
                }
            } else {
                for k in 0..l {
                    self.g_bar[k] += c_i * q_i_full[k] as f64;
                }
            }
        }

        if uj != self.is_upper_bound(j) {
            let q_j_full = self.q.get_q(j, l).to_vec();
            if uj {
                for k in 0..l {
                    self.g_bar[k] -= c_j * q_j_full[k] as f64;
                }
            } else {
                for k in 0..l {
                    self.g_bar[k] += c_j * q_j_full[k] as f64;
                }
            }
        }
    }

    // ─── Shrinking ──────────────────────────────────────────────────

    fn do_shrinking(&mut self) {
        match self.variant {
            SolverVariant::Standard => self.do_shrinking_standard(),
            SolverVariant::Nu => self.do_shrinking_nu(),
        }
    }

    fn be_shrunk_standard(&self, i: usize, gmax1: f64, gmax2: f64) -> bool {
        if self.is_upper_bound(i) {
            if self.y[i] == 1 {
                -self.g[i] > gmax1
            } else {
                -self.g[i] > gmax2
            }
        } else if self.is_lower_bound(i) {
            if self.y[i] == 1 {
                self.g[i] > gmax2
            } else {
                self.g[i] > gmax1
            }
        } else {
            false
        }
    }

    fn do_shrinking_standard(&mut self) {
        let mut gmax1 = -INF;
        let mut gmax2 = -INF;

        for i in 0..self.active_size {
            if self.y[i] == 1 {
                if !self.is_upper_bound(i) && -self.g[i] >= gmax1 {
                    gmax1 = -self.g[i];
                }
                if !self.is_lower_bound(i) && self.g[i] >= gmax2 {
                    gmax2 = self.g[i];
                }
            } else {
                if !self.is_upper_bound(i) && -self.g[i] >= gmax2 {
                    gmax2 = -self.g[i];
                }
                if !self.is_lower_bound(i) && self.g[i] >= gmax1 {
                    gmax1 = self.g[i];
                }
            }
        }

        if !self.unshrink && gmax1 + gmax2 <= self.eps * 10.0 {
            self.unshrink = true;
            self.reconstruct_gradient();
            self.active_size = self.l;
        }

        let mut i = 0;
        while i < self.active_size {
            if self.be_shrunk_standard(i, gmax1, gmax2) {
                self.active_size -= 1;
                while self.active_size > i {
                    if !self.be_shrunk_standard(self.active_size, gmax1, gmax2) {
                        self.swap_index(i, self.active_size);
                        break;
                    }
                    self.active_size -= 1;
                }
            }
            i += 1;
        }
    }

    fn be_shrunk_nu(&self, i: usize, gmax1: f64, gmax2: f64, gmax3: f64, gmax4: f64) -> bool {
        if self.is_upper_bound(i) {
            if self.y[i] == 1 {
                -self.g[i] > gmax1
            } else {
                -self.g[i] > gmax4
            }
        } else if self.is_lower_bound(i) {
            if self.y[i] == 1 {
                self.g[i] > gmax2
            } else {
                self.g[i] > gmax3
            }
        } else {
            false
        }
    }

    fn do_shrinking_nu(&mut self) {
        let mut gmax1 = -INF;
        let mut gmax2 = -INF;
        let mut gmax3 = -INF;
        let mut gmax4 = -INF;

        for i in 0..self.active_size {
            if !self.is_upper_bound(i) {
                if self.y[i] == 1 {
                    if -self.g[i] > gmax1 {
                        gmax1 = -self.g[i];
                    }
                } else if -self.g[i] > gmax4 {
                    gmax4 = -self.g[i];
                }
            }
            if !self.is_lower_bound(i) {
                if self.y[i] == 1 {
                    if self.g[i] > gmax2 {
                        gmax2 = self.g[i];
                    }
                } else if self.g[i] > gmax3 {
                    gmax3 = self.g[i];
                }
            }
        }

        if !self.unshrink && f64::max(gmax1 + gmax2, gmax3 + gmax4) <= self.eps * 10.0 {
            self.unshrink = true;
            self.reconstruct_gradient();
            self.active_size = self.l;
        }

        let mut i = 0;
        while i < self.active_size {
            if self.be_shrunk_nu(i, gmax1, gmax2, gmax3, gmax4) {
                self.active_size -= 1;
                while self.active_size > i {
                    if !self.be_shrunk_nu(self.active_size, gmax1, gmax2, gmax3, gmax4) {
                        self.swap_index(i, self.active_size);
                        break;
                    }
                    self.active_size -= 1;
                }
            }
            i += 1;
        }
    }

    // ─── Rho calculation ────────────────────────────────────────────

    fn calculate_rho(&self) -> (f64, f64) {
        match self.variant {
            SolverVariant::Standard => (self.calculate_rho_standard(), 0.0),
            SolverVariant::Nu => self.calculate_rho_nu(),
        }
    }

    fn calculate_rho_standard(&self) -> f64 {
        let mut nr_free = 0;
        let mut ub = INF;
        let mut lb = -INF;
        let mut sum_free = 0.0;

        for i in 0..self.active_size {
            let yg = self.y[i] as f64 * self.g[i];

            if self.is_upper_bound(i) {
                if self.y[i] == -1 {
                    ub = ub.min(yg);
                } else {
                    lb = lb.max(yg);
                }
            } else if self.is_lower_bound(i) {
                if self.y[i] == 1 {
                    ub = ub.min(yg);
                } else {
                    lb = lb.max(yg);
                }
            } else {
                nr_free += 1;
                sum_free += yg;
            }
        }

        if nr_free > 0 {
            sum_free / nr_free as f64
        } else {
            (ub + lb) / 2.0
        }
    }

    fn calculate_rho_nu(&self) -> (f64, f64) {
        let mut nr_free1 = 0;
        let mut nr_free2 = 0;
        let mut ub1 = INF;
        let mut ub2 = INF;
        let mut lb1 = -INF;
        let mut lb2 = -INF;
        let mut sum_free1 = 0.0;
        let mut sum_free2 = 0.0;

        for i in 0..self.active_size {
            if self.y[i] == 1 {
                if self.is_upper_bound(i) {
                    lb1 = lb1.max(self.g[i]);
                } else if self.is_lower_bound(i) {
                    ub1 = ub1.min(self.g[i]);
                } else {
                    nr_free1 += 1;
                    sum_free1 += self.g[i];
                }
            } else if self.is_upper_bound(i) {
                lb2 = lb2.max(self.g[i]);
            } else if self.is_lower_bound(i) {
                ub2 = ub2.min(self.g[i]);
            } else {
                nr_free2 += 1;
                sum_free2 += self.g[i];
            }
        }

        let r1 = if nr_free1 > 0 {
            sum_free1 / nr_free1 as f64
        } else {
            (ub1 + lb1) / 2.0
        };

        let r2 = if nr_free2 > 0 {
            sum_free2 / nr_free2 as f64
        } else {
            (ub2 + lb2) / 2.0
        };

        let rho = (r1 - r2) / 2.0;
        let r = (r1 + r2) / 2.0;
        (rho, r)
    }
}
