//! Fluent construction for [`SvmParameter`].
//!
//! `SvmParameterBuilder` is an ergonomic layer over [`SvmParameter`]. It keeps
//! the same defaults as [`SvmParameter::default`] and delegates construction-time,
//! data-independent validation to [`SvmParameter::validate`]. Data-dependent
//! checks, including problem shape, precomputed-kernel rows, and ν-SVC class
//! feasibility, remain in [`crate::types::check_parameter`].
//!
//! # Example
//!
//! ```
//! use libsvm_rs::{KernelType, SvmParameterBuilder, SvmType};
//!
//! let param = SvmParameterBuilder::new()
//!     .svm_type(SvmType::CSvc)
//!     .kernel_type(KernelType::Rbf)
//!     .gamma(1.0 / 13.0)
//!     .c(1.0)
//!     .build()?;
//! # Ok::<(), libsvm_rs::SvmError>(())
//! ```

#![cfg_attr(
    test,
    expect(
        clippy::unwrap_used,
        reason = "Checked access is not yet done in this module; remove this line when it is"
    )
)]

use crate::error::SvmError;
use crate::types::{KernelType, SvmParameter, SvmType};

/// Fluent builder for a validated [`SvmParameter`].
///
/// Defaults are identical to [`SvmParameter::default`]. [`build`](Self::build)
/// constructs the parameter and calls the existing [`SvmParameter::validate`]
/// method; data-dependent validation remains the responsibility of
/// [`crate::types::check_parameter`].
///
/// # Example
///
/// ```
/// use libsvm_rs::SvmParameterBuilder;
/// let param = SvmParameterBuilder::new().c(2.0).build()?;
/// assert_eq!(param.c.to_bits(), 2.0_f64.to_bits());
/// Ok::<(), libsvm_rs::SvmError>(())
/// ```
#[derive(Debug, Clone)]
#[must_use]
pub struct SvmParameterBuilder {
    param: SvmParameter,
}

impl Default for SvmParameterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SvmParameterBuilder {
    /// Create a builder using LIBSVM defaults.
    ///
    /// # Example
    ///
    /// ```
    /// use libsvm_rs::SvmParameterBuilder;
    /// let param = SvmParameterBuilder::new().build()?;
    /// assert_eq!(param, libsvm_rs::SvmParameter::default());
    /// Ok::<(), libsvm_rs::SvmError>(())
    /// ```
    pub fn new() -> Self {
        Self {
            param: SvmParameter::default(),
        }
    }

    /// Set the SVM formulation type. Default: [`SvmType::CSvc`].
    ///
    /// # Example
    ///
    /// ```
    /// use libsvm_rs::SvmParameterBuilder;
    /// let param = SvmParameterBuilder::new().svm_type(libsvm_rs::SvmType::NuSvc).build()?;
    /// assert_eq!(param.svm_type, libsvm_rs::SvmType::NuSvc);
    /// Ok::<(), libsvm_rs::SvmError>(())
    /// ```
    pub fn svm_type(mut self, svm_type: SvmType) -> Self {
        self.param.svm_type = svm_type;
        self
    }

    /// Set the kernel function type. Default: [`KernelType::Rbf`].
    ///
    /// # Example
    ///
    /// ```
    /// use libsvm_rs::SvmParameterBuilder;
    /// let param = SvmParameterBuilder::new().kernel_type(libsvm_rs::KernelType::Linear).build()?;
    /// assert_eq!(param.kernel_type, libsvm_rs::KernelType::Linear);
    /// Ok::<(), libsvm_rs::SvmError>(())
    /// ```
    pub fn kernel_type(mut self, kernel_type: KernelType) -> Self {
        self.param.kernel_type = kernel_type;
        self
    }

    /// Set the polynomial kernel degree. Default: `3`.
    ///
    /// # Example
    ///
    /// ```
    /// use libsvm_rs::SvmParameterBuilder;
    /// let param = SvmParameterBuilder::new().degree(2).build()?;
    /// assert_eq!(param.degree, 2);
    /// Ok::<(), libsvm_rs::SvmError>(())
    /// ```
    pub fn degree(mut self, degree: i32) -> Self {
        self.param.degree = degree;
        self
    }

    /// Set γ for RBF, polynomial, and sigmoid kernels. Default: `0.0`
    /// (interpreted during training as `1 / num_features`).
    ///
    /// # Example
    ///
    /// ```
    /// use libsvm_rs::SvmParameterBuilder;
    /// let param = SvmParameterBuilder::new().gamma(0.5).build()?;
    /// assert_eq!(param.gamma.to_bits(), 0.5_f64.to_bits());
    /// Ok::<(), libsvm_rs::SvmError>(())
    /// ```
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.param.gamma = gamma;
        self
    }

    /// Set the independent term in polynomial and sigmoid kernels. Default: `0.0`.
    ///
    /// # Example
    ///
    /// ```
    /// use libsvm_rs::SvmParameterBuilder;
    /// let param = SvmParameterBuilder::new().coef0(1.0).build()?;
    /// assert_eq!(param.coef0.to_bits(), 1.0_f64.to_bits());
    /// Ok::<(), libsvm_rs::SvmError>(())
    /// ```
    pub fn coef0(mut self, coef0: f64) -> Self {
        self.param.coef0 = coef0;
        self
    }

    /// Set the cost parameter C. Default: `1.0`.
    ///
    /// # Example
    ///
    /// ```
    /// use libsvm_rs::SvmParameterBuilder;
    /// let param = SvmParameterBuilder::new().c(2.0).build()?;
    /// assert_eq!(param.c.to_bits(), 2.0_f64.to_bits());
    /// Ok::<(), libsvm_rs::SvmError>(())
    /// ```
    pub fn c(mut self, c: f64) -> Self {
        self.param.c = c;
        self
    }

    /// Set ν for ν-SVC, one-class SVM, and ν-SVR. Default: `0.5`.
    ///
    /// # Example
    ///
    /// ```
    /// use libsvm_rs::SvmParameterBuilder;
    /// let param = SvmParameterBuilder::new().nu(0.25).build()?;
    /// assert_eq!(param.nu.to_bits(), 0.25_f64.to_bits());
    /// Ok::<(), libsvm_rs::SvmError>(())
    /// ```
    pub fn nu(mut self, nu: f64) -> Self {
        self.param.nu = nu;
        self
    }

    /// Set ε in the ε-insensitive loss function for ε-SVR. Default: `0.1`.
    ///
    /// # Example
    ///
    /// ```
    /// use libsvm_rs::SvmParameterBuilder;
    /// let param = SvmParameterBuilder::new().p(0.2).build()?;
    /// assert_eq!(param.p.to_bits(), 0.2_f64.to_bits());
    /// Ok::<(), libsvm_rs::SvmError>(())
    /// ```
    pub fn p(mut self, p: f64) -> Self {
        self.param.p = p;
        self
    }

    /// Set cache memory size in MB. Default: `100.0`.
    ///
    /// # Example
    ///
    /// ```
    /// use libsvm_rs::SvmParameterBuilder;
    /// let param = SvmParameterBuilder::new().cache_size(16.0).build()?;
    /// assert_eq!(param.cache_size.to_bits(), 16.0_f64.to_bits());
    /// Ok::<(), libsvm_rs::SvmError>(())
    /// ```
    pub fn cache_size(mut self, cache_size: f64) -> Self {
        self.param.cache_size = cache_size;
        self
    }

    /// Set solver stopping tolerance. Default: `0.001`.
    ///
    /// # Example
    ///
    /// ```
    /// use libsvm_rs::SvmParameterBuilder;
    /// let param = SvmParameterBuilder::new().eps(0.0001).build()?;
    /// assert_eq!(param.eps.to_bits(), 0.0001_f64.to_bits());
    /// Ok::<(), libsvm_rs::SvmError>(())
    /// ```
    pub fn eps(mut self, eps: f64) -> Self {
        self.param.eps = eps;
        self
    }

    /// Enable or disable the shrinking heuristic. Default: `true`.
    ///
    /// # Example
    ///
    /// ```
    /// use libsvm_rs::SvmParameterBuilder;
    /// let param = SvmParameterBuilder::new().shrinking(false).build()?;
    /// assert!(!param.shrinking);
    /// Ok::<(), libsvm_rs::SvmError>(())
    /// ```
    pub fn shrinking(mut self, shrinking: bool) -> Self {
        self.param.shrinking = shrinking;
        self
    }

    /// Enable or disable probability-estimate training. Default: `false`.
    ///
    /// # Example
    ///
    /// ```
    /// use libsvm_rs::SvmParameterBuilder;
    /// let param = SvmParameterBuilder::new().probability(true).build()?;
    /// assert!(param.probability);
    /// Ok::<(), libsvm_rs::SvmError>(())
    /// ```
    pub fn probability(mut self, probability: bool) -> Self {
        self.param.probability = probability;
        self
    }

    /// Append one per-class weight override `(class_label, weight)`. Default: empty.
    ///
    /// # Example
    ///
    /// ```
    /// use libsvm_rs::SvmParameterBuilder;
    /// let param = SvmParameterBuilder::new().weight(1, 2.0).build()?;
    /// assert_eq!(param.weight, vec![(1, 2.0)]);
    /// Ok::<(), libsvm_rs::SvmError>(())
    /// ```
    pub fn weight(mut self, label: i32, weight: f64) -> Self {
        self.param.weight.push((label, weight));
        self
    }

    /// Replace all per-class weight overrides. Default: empty.
    ///
    /// # Example
    ///
    /// ```
    /// use libsvm_rs::SvmParameterBuilder;
    /// let param = SvmParameterBuilder::new().weight(1, 2.0).weights(vec![(-1, 3.0)]).build()?;
    /// assert_eq!(param.weight, vec![(-1, 3.0)]);
    /// Ok::<(), libsvm_rs::SvmError>(())
    /// ```
    pub fn weights(mut self, weights: Vec<(i32, f64)>) -> Self {
        self.param.weight = weights;
        self
    }

    /// Construct and validate an [`SvmParameter`].
    ///
    /// This calls the existing [`SvmParameter::validate`] method and does not
    /// duplicate validation rules. Data-dependent checks stay in
    /// [`crate::types::check_parameter`].
    ///
    /// # Errors
    ///
    /// Returns [`SvmError::InvalidParameter`] when a parameter fails
    /// [`SvmParameter::validate`].
    ///
    /// # Example
    ///
    /// ```
    /// use libsvm_rs::SvmParameterBuilder;
    /// let valid = SvmParameterBuilder::new().build()?;
    /// assert!(valid.validate().is_ok());
    /// assert!(SvmParameterBuilder::new().eps(0.0).build().is_err());
    /// Ok::<(), libsvm_rs::SvmError>(())
    /// ```
    pub fn build(self) -> Result<SvmParameter, SvmError> {
        self.param.validate()?;
        Ok(self.param)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_method_build_equals_parameter_default() {
        assert_eq!(
            SvmParameterBuilder::new().build().unwrap(),
            SvmParameter::default()
        );
    }

    #[test]
    #[expect(
        clippy::field_reassign_with_default,
        reason = "A test assigns fields to match the builder"
    )]
    fn happy_path_equals_field_assignment() {
        let built = SvmParameterBuilder::new()
            .svm_type(SvmType::EpsilonSvr)
            .kernel_type(KernelType::Sigmoid)
            .degree(2)
            .gamma(0.25)
            .coef0(1.5)
            .c(2.0)
            .nu(0.25)
            .p(0.2)
            .cache_size(256.0)
            .eps(0.0001)
            .shrinking(false)
            .probability(true)
            .weight(1, 3.0)
            .weight(-1, 0.5)
            .build()
            .unwrap();

        let mut assigned = SvmParameter::default();
        assigned.svm_type = SvmType::EpsilonSvr;
        assigned.kernel_type = KernelType::Sigmoid;
        assigned.degree = 2;
        assigned.gamma = 0.25;
        assigned.coef0 = 1.5;
        assigned.c = 2.0;
        assigned.nu = 0.25;
        assigned.p = 0.2;
        assigned.cache_size = 256.0;
        assigned.eps = 0.0001;
        assigned.shrinking = false;
        assigned.probability = true;
        assigned.weight = vec![(1, 3.0), (-1, 0.5)];

        assert_eq!(built, assigned);
    }

    #[test]
    fn weights_replaces_weight_list() {
        let built = SvmParameterBuilder::new()
            .weight(1, 2.0)
            .weights(vec![(3, 4.0), (5, 6.0)])
            .build()
            .unwrap();

        assert_eq!(built.weight, vec![(3, 4.0), (5, 6.0)]);
    }

    #[test]
    fn negative_gamma_rejected_by_build() {
        assert!(matches!(
            SvmParameterBuilder::new().gamma(-1.0).build(),
            Err(SvmError::InvalidParameter(_))
        ));
    }

    #[test]
    fn non_positive_eps_rejected_by_build() {
        assert!(matches!(
            SvmParameterBuilder::new().eps(0.0).build(),
            Err(SvmError::InvalidParameter(_))
        ));
    }

    #[test]
    fn non_positive_cache_size_rejected_by_build() {
        assert!(matches!(
            SvmParameterBuilder::new().cache_size(0.0).build(),
            Err(SvmError::InvalidParameter(_))
        ));
    }

    #[test]
    fn negative_polynomial_degree_rejected_by_build() {
        assert!(matches!(
            SvmParameterBuilder::new()
                .kernel_type(KernelType::Polynomial)
                .degree(-1)
                .build(),
            Err(SvmError::InvalidParameter(_))
        ));
    }
}
