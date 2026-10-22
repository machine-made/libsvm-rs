//! I/O routines for LIBSVM problem and model files.
//!
//! The file formats match the original LIBSVM text formats for cross-tool
//! interoperability, but the loaders are written for untrusted input by
//! default. They enforce [`LoadOptions`] caps, reject embedded NUL bytes,
//! reject non-ascending feature indices, and return structured [`SvmError`]
//! values for malformed input within those caps.
//!
//! Model loading performs additional structural checks before allocating support
//! vector storage. Header fields such as `nr_class`, `total_sv`, `rho`, `label`,
//! `nr_sv`, `probA`, `probB`, and one-class probability-density marks must be
//! shape-consistent when present. Precomputed-kernel support-vector rows must
//! begin with `0:sample_serial_number`.
//!
//! These checks validate text structure and resource bounds. They do not prove
//! that a model is statistically meaningful, was trained from a particular
//! dataset, or is cryptographically authentic.

use std::io::{BufRead, Write};
use std::path::Path;

use crate::error::SvmError;
use crate::types::*;
use crate::util::parse_feature_index;
use crate::util::MAX_FEATURE_INDEX;

// ─── C-compatible %g formatting ─────────────────────────────────────
//
// C's printf `%.Pg` format strips trailing zeros and picks fixed vs.
// scientific notation based on the exponent. Rust has no built-in
// equivalent, so we replicate the POSIX specification:
//   - Use scientific if exponent < -4 or exponent >= precision
//   - Otherwise use fixed notation
//   - Strip trailing zeros (and trailing decimal point)

use std::fmt;

/// Formats `f64` like C's `%.17g` (or any precision).
struct Gfmt {
    value: f64,
    precision: usize,
}

impl Gfmt {
    fn new(value: f64, precision: usize) -> Self {
        Self { value, precision }
    }
}

impl fmt::Display for Gfmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let v = self.value;
        let p = self.precision;

        if !v.is_finite() {
            return write!(f, "{}", v); // inf, -inf, NaN
        }

        if v == 0.0 {
            // Preserve sign of -0.0
            if v.is_sign_negative() {
                return write!(f, "-0");
            }
            return write!(f, "0");
        }

        // Compute the exponent (floor of log10(|v|))
        let abs_v = v.abs();
        let exp = abs_v.log10().floor() as i32;

        if exp < -4 || exp >= p as i32 {
            // Use scientific notation
            let s = format!("{:.prec$e}", v, prec = p.saturating_sub(1));
            // Rust uses 'e', C uses 'e'. Strip trailing zeros in mantissa.
            // C zero-pads exponent to at least 2 digits (e-05 not e-5).
            if let Some((mantissa, exponent)) = s.split_once('e') {
                let mantissa = mantissa.trim_end_matches('0').trim_end_matches('.');
                // Parse exponent, reformat with at least 2 digits
                let exp_val: i32 = exponent.parse().unwrap_or(0);
                let exp_str = if exp_val < 0 {
                    format!("-{:02}", -exp_val)
                } else {
                    format!("+{:02}", exp_val)
                };
                write!(f, "{}e{}", mantissa, exp_str)
            } else {
                write!(f, "{}", s)
            }
        } else {
            // Use fixed notation. Number of decimal places = precision - (exp + 1)
            let decimal_places = if exp >= 0 {
                p.saturating_sub((exp + 1) as usize)
            } else {
                p + (-1 - exp) as usize
            };
            let s = format!("{:.prec$}", v, prec = decimal_places);
            let s = s.trim_end_matches('0').trim_end_matches('.');
            write!(f, "{}", s)
        }
    }
}

/// Format like C's `%.17g`
fn fmt_17g(v: f64) -> Gfmt {
    Gfmt::new(v, 17)
}

/// Format like C's `%.8g`
fn fmt_8g(v: f64) -> Gfmt {
    Gfmt::new(v, 8)
}

/// Format a float like C's `%g` (6 significant digits).
pub fn format_g(v: f64) -> String {
    format!("{}", Gfmt::new(v, 6))
}

/// Format a float like C's `%.17g` (17 significant digits).
pub fn format_17g(v: f64) -> String {
    format!("{}", Gfmt::new(v, 17))
}

// ─── String tables matching original LIBSVM ──────────────────────────

const SVM_TYPE_TABLE: &[&str] = &["c_svc", "nu_svc", "one_class", "epsilon_svr", "nu_svr"];
const KERNEL_TYPE_TABLE: &[&str] = &["linear", "polynomial", "rbf", "sigmoid", "precomputed"];

fn svm_type_to_str(t: SvmType) -> &'static str {
    SVM_TYPE_TABLE[t as usize]
}

fn kernel_type_to_str(t: KernelType) -> &'static str {
    KERNEL_TYPE_TABLE[t as usize]
}

fn str_to_svm_type(s: &str) -> Option<SvmType> {
    match s {
        "c_svc" => Some(SvmType::CSvc),
        "nu_svc" => Some(SvmType::NuSvc),
        "one_class" => Some(SvmType::OneClass),
        "epsilon_svr" => Some(SvmType::EpsilonSvr),
        "nu_svr" => Some(SvmType::NuSvr),
        _ => None,
    }
}

fn str_to_kernel_type(s: &str) -> Option<KernelType> {
    match s {
        "linear" => Some(KernelType::Linear),
        "polynomial" => Some(KernelType::Polynomial),
        "rbf" => Some(KernelType::Rbf),
        "sigmoid" => Some(KernelType::Sigmoid),
        "precomputed" => Some(KernelType::Precomputed),
        _ => None,
    }
}

// ─── Load options ────────────────────────────────────────────────────

/// Resource caps applied while reading LIBSVM problem and model files.
///
/// LIBSVM text formats are linear in file size, but individual fields (e.g.
/// `total_sv`, feature indices, per-line token counts) can be used by a
/// malicious file to trigger disproportionate allocation or CPU work. The
/// [`Default`] impl returns defaults tuned for **untrusted input**:
///
/// | Field               | Default        |
/// |---------------------|----------------|
/// | `max_bytes`         | 64 MiB         |
/// | `max_line_len`      | 1 MiB          |
/// | `max_sv`            | 10 000 000     |
/// | `max_nr_class`      | 65 535         |
/// | `max_feature_index` | 10 000 000     |
///
/// If you know the input is trusted (produced locally, read from an
/// attestation-protected location, etc.) and you need to handle inputs
/// larger than the defaults, call [`LoadOptions::trusted_input`] for an
/// "unlimited" profile, or override specific fields:
///
/// ```ignore
/// use libsvm_rs::io::{load_problem_from_reader_with_options, LoadOptions};
///
/// let opts = LoadOptions {
///     max_bytes: 1 << 30, // 1 GiB for a trusted bulk file
///     ..LoadOptions::default()
/// };
/// let problem = load_problem_from_reader_with_options(reader, &opts)?;
/// ```
///
/// Exceeding any cap returns an [`SvmError::ParseError`] (problem files) or
/// [`SvmError::ModelFormatError`] (model files) with a message identifying
/// the field that tripped the limit.
///
/// The default profile is intended to be panic-free for malformed text input
/// within these caps. It is still a parser contract, not a semantic model
/// audit: it does not verify that a loaded model came from a specific training
/// set or that its predictions are appropriate for a deployment.
#[derive(Debug, Clone, Copy)]
pub struct LoadOptions {
    /// Maximum total number of bytes read from the source.
    pub max_bytes: u64,
    /// Maximum number of bytes in a single line (excluding the terminator).
    pub max_line_len: usize,
    /// Maximum value accepted for the `total_sv` header field in model files.
    pub max_sv: usize,
    /// Maximum value accepted for the `nr_class` header field in model files.
    pub max_nr_class: usize,
    /// Maximum feature index accepted in problem or model feature lists.
    pub max_feature_index: i32,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            max_bytes: 64 * 1024 * 1024,
            max_line_len: 1024 * 1024,
            max_sv: MAX_TOTAL_SV,
            max_nr_class: MAX_NR_CLASS,
            max_feature_index: MAX_FEATURE_INDEX,
        }
    }
}

impl LoadOptions {
    /// Options with all caps set to the type maximum.
    ///
    /// Use **only** for input that is fully trusted. Operating with
    /// unlimited caps removes the first line of defense against malformed
    /// or adversarial model / problem files. Module-level hard caps still
    /// apply to `nr_class` and `total_sv`.
    pub fn trusted_input() -> Self {
        Self {
            max_bytes: u64::MAX,
            max_line_len: usize::MAX,
            max_sv: usize::MAX,
            max_nr_class: usize::MAX,
            max_feature_index: i32::MAX,
        }
    }
}

/// Read one line from `reader`, enforcing byte and line-length caps.
///
/// Returns `Ok(None)` on clean EOF. Updates `bytes_read` by the number of
/// bytes actually consumed (including the line terminator).
///
/// Uses [`BufRead::fill_buf`] / [`BufRead::consume`] to pull at most a
/// buffer's worth at a time and check both caps before committing bytes
/// to the output buffer. A pathological unbounded line cannot grow the
/// output buffer past `max_line_len` — the error fires before the next
/// slice copy would push past the cap.
fn read_line_capped(
    reader: &mut dyn BufRead,
    bytes_read: &mut u64,
    max_bytes: u64,
    max_line_len: usize,
) -> std::io::Result<Option<String>> {
    // One byte of slack on the per-line cap to accommodate `\r\n` line
    // endings: `max_line_len` describes *content* length, so a line with
    // the full allowed content plus CRLF consumes `max_line_len + 2` bytes
    // total, of which one (the \r) would otherwise push us past the cap
    // before the final `\n` arrives.
    let per_line_raw_cap: u64 = (max_line_len as u64).saturating_add(1);

    let mut buf: Vec<u8> = Vec::new();
    let mut found_newline = false;

    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            break;
        }

        let take_n = match available.iter().position(|&b| b == b'\n') {
            Some(pos) => pos + 1, // include the newline
            None => available.len(),
        };
        let ends_with_newline = take_n > 0 && available[take_n - 1] == b'\n';

        // Byte cap.
        let new_bytes_read = bytes_read.saturating_add(take_n as u64);
        if new_bytes_read > max_bytes {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("input exceeds max_bytes limit ({})", max_bytes),
            ));
        }

        // Per-line cap. Content bytes excludes the trailing `\n` (if any);
        // we allow one byte of slack for a possible `\r` preceding it.
        let prospective_len = (buf.len() as u64).saturating_add(take_n as u64);
        let content_bytes = if ends_with_newline {
            prospective_len - 1
        } else {
            prospective_len
        };
        if content_bytes > per_line_raw_cap {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("line length exceeds max_line_len limit ({})", max_line_len),
            ));
        }

        // NUL bytes have no legal use in LIBSVM text files.
        if available[..take_n].contains(&0) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "unexpected NUL byte in text input".to_string(),
            ));
        }

        buf.extend_from_slice(&available[..take_n]);
        reader.consume(take_n);
        *bytes_read = new_bytes_read;

        if ends_with_newline {
            found_newline = true;
            break;
        }
    }

    if buf.is_empty() && !found_newline {
        return Ok(None);
    }

    let line = String::from_utf8(buf)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
    Ok(Some(line))
}

// ─── Problem file I/O ────────────────────────────────────────────────

/// Load an SVM problem from a file in LIBSVM sparse format.
///
/// Format: `<label> <index1>:<value1> <index2>:<value2> ...`
///
/// Uses [`LoadOptions::default`] — appropriate for untrusted input. For
/// trusted bulk files exceeding the defaults, use
/// [`load_problem_from_reader_with_options`] with a custom [`LoadOptions`].
///
/// Validates:
///
/// - total file size and per-line length,
/// - embedded NUL byte absence,
/// - parseable labels and feature values,
/// - `index:value` token shape,
/// - ascending feature indices,
/// - [`LoadOptions::max_feature_index`].
///
/// This loader does not validate statistical quality, feature normalization, or
/// whether labels are appropriate for a particular SVM formulation. Malformed
/// text input within the configured caps is returned as [`SvmError`] rather
/// than panicking.
///
/// ### Complexity
///
/// Linear in file size for parsing and `O(n · d)` memory where `n` is the
/// number of instances and `d` is the average number of non-zero features
/// per instance. No per-row allocation is driven by an untrusted header.
pub fn load_problem(path: &Path) -> Result<SvmProblem, SvmError> {
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    load_problem_from_reader(reader)
}

/// Load an SVM problem from any buffered reader.
///
/// Uses [`LoadOptions::default`]. See [`load_problem`] for the validation
/// contract and non-goals.
pub fn load_problem_from_reader(reader: impl BufRead) -> Result<SvmProblem, SvmError> {
    load_problem_from_reader_with_options(reader, &LoadOptions::default())
}

/// Load an SVM problem from any buffered reader, with explicit resource caps.
///
/// See [`LoadOptions`] for the meaning of each cap and for defaults tuned for
/// untrusted input. This function has the same validation contract as
/// [`load_problem`], with caller-supplied caps.
pub fn load_problem_from_reader_with_options(
    mut reader: impl BufRead,
    options: &LoadOptions,
) -> Result<SvmProblem, SvmError> {
    let mut labels = Vec::new();
    let mut instances = Vec::new();
    let mut bytes_read: u64 = 0;
    let mut line_idx: usize = 0;

    while let Some(raw) = read_line_capped(
        &mut reader,
        &mut bytes_read,
        options.max_bytes,
        options.max_line_len,
    )? {
        let line_num = line_idx + 1;
        line_idx += 1;
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }

        let mut parts = line.split_whitespace();

        // Parse label
        let label_str = parts.next().ok_or_else(|| SvmError::ParseError {
            line: line_num,
            message: "missing label".into(),
        })?;
        let label: f64 = label_str.parse().map_err(|_| SvmError::ParseError {
            line: line_num,
            message: format!("invalid label: {}", label_str),
        })?;

        // Parse features (must be in ascending index order).
        let mut nodes = Vec::new();
        let mut prev_index: i32 = 0;
        for token in parts {
            let (idx_str, val_str) = token.split_once(':').ok_or_else(|| SvmError::ParseError {
                line: line_num,
                message: format!("expected index:value, got: {}", token),
            })?;
            let index: i32 =
                parse_feature_index_problem_line(line_num, idx_str, options.max_feature_index)?;

            if !nodes.is_empty() && index <= prev_index {
                return Err(SvmError::ParseError {
                    line: line_num,
                    message: format!(
                        "feature indices must be ascending: {} follows {}",
                        index, prev_index
                    ),
                });
            }
            let value: f64 = val_str.parse().map_err(|_| SvmError::ParseError {
                line: line_num,
                message: format!("invalid value: {}", val_str),
            })?;
            prev_index = index;
            nodes.push(SvmNode { index, value });
        }

        labels.push(label);
        instances.push(nodes);
    }

    Ok(SvmProblem { labels, instances })
}

// ─── Model file I/O ──────────────────────────────────────────────────

const MAX_NR_CLASS: usize = 65535;
const MAX_TOTAL_SV: usize = 10_000_000;

/// Save an SVM model to a file in the original LIBSVM format.
pub fn save_model(path: &Path, model: &SvmModel) -> Result<(), SvmError> {
    let file = std::fs::File::create(path)?;
    let writer = std::io::BufWriter::new(file);
    save_model_to_writer(writer, model)
}

/// Save an SVM model to any writer.
///
/// This is the writer-backed equivalent of LIBSVM's `svm_save_model`, useful
/// for in-memory buffers and tests.
///
/// ```
/// use libsvm_rs::io::{load_model_from_reader, save_model_to_writer};
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
/// let model = svm_train(&problem, &param);
///
/// let mut bytes = Vec::new();
/// save_model_to_writer(&mut bytes, &model)?;
/// let loaded = load_model_from_reader(std::io::Cursor::new(bytes))?;
/// assert_eq!(loaded.svm_type(), SvmType::CSvc);
/// # Ok::<(), libsvm_rs::SvmError>(())
/// ```
pub fn save_model_to_writer(mut w: impl Write, model: &SvmModel) -> Result<(), SvmError> {
    let param = &model.param;

    writeln!(w, "svm_type {}", svm_type_to_str(param.svm_type))?;
    writeln!(w, "kernel_type {}", kernel_type_to_str(param.kernel_type))?;

    if param.kernel_type == KernelType::Polynomial {
        writeln!(w, "degree {}", param.degree)?;
    }
    if matches!(
        param.kernel_type,
        KernelType::Polynomial | KernelType::Rbf | KernelType::Sigmoid
    ) {
        writeln!(w, "gamma {}", fmt_17g(param.gamma))?;
    }
    if matches!(
        param.kernel_type,
        KernelType::Polynomial | KernelType::Sigmoid
    ) {
        writeln!(w, "coef0 {}", fmt_17g(param.coef0))?;
    }

    let nr_class = model.nr_class;
    writeln!(w, "nr_class {}", nr_class)?;
    writeln!(w, "total_sv {}", model.sv.len())?;

    // rho
    write!(w, "rho")?;
    for r in &model.rho {
        write!(w, " {}", fmt_17g(*r))?;
    }
    writeln!(w)?;

    // label (classification only)
    if !model.label.is_empty() {
        write!(w, "label")?;
        for l in &model.label {
            write!(w, " {}", l)?;
        }
        writeln!(w)?;
    }

    // probA
    if !model.prob_a.is_empty() {
        write!(w, "probA")?;
        for v in &model.prob_a {
            write!(w, " {}", fmt_17g(*v))?;
        }
        writeln!(w)?;
    }

    // probB
    if !model.prob_b.is_empty() {
        write!(w, "probB")?;
        for v in &model.prob_b {
            write!(w, " {}", fmt_17g(*v))?;
        }
        writeln!(w)?;
    }

    // prob_density_marks (one-class)
    if !model.prob_density_marks.is_empty() {
        write!(w, "prob_density_marks")?;
        for v in &model.prob_density_marks {
            write!(w, " {}", fmt_17g(*v))?;
        }
        writeln!(w)?;
    }

    // nr_sv
    if !model.n_sv.is_empty() {
        write!(w, "nr_sv")?;
        for n in &model.n_sv {
            write!(w, " {}", n)?;
        }
        writeln!(w)?;
    }

    // SV section
    writeln!(w, "SV")?;
    let num_sv = model.sv.len();
    let num_coef_rows = model.sv_coef.len(); // nr_class - 1

    for i in 0..num_sv {
        // sv_coef columns for this SV: %.17g
        for j in 0..num_coef_rows {
            write!(w, "{} ", fmt_17g(model.sv_coef[j][i]))?;
        }
        // sparse features: %.8g
        if model.param.kernel_type == KernelType::Precomputed {
            if let Some(node) = model.sv[i].first() {
                write!(w, "0:{} ", node.value as i32)?;
            }
        } else {
            for node in &model.sv[i] {
                write!(w, "{}:{} ", node.index, fmt_8g(node.value))?;
            }
        }
        writeln!(w)?;
    }

    Ok(())
}

/// Load an SVM model from a file in the original LIBSVM format.
///
/// Uses [`LoadOptions::default`] — appropriate for untrusted input.
///
/// Validates:
///
/// - total file size and per-line length,
/// - embedded NUL byte absence,
/// - known `svm_type` and `kernel_type` values,
/// - `nr_class` and `total_sv` caps,
/// - `nr_class >= 2`,
/// - `rho` length for classification, one-class, and regression models,
/// - `label` and `nr_sv` length when present,
/// - `sum(nr_sv) == total_sv` when `nr_sv` is present,
/// - `probA` / `probB` decision-function counts when present,
/// - one-class-only `prob_density_marks`,
/// - support-vector feature token shape, ascending feature indices, and
///   [`LoadOptions::max_feature_index`],
/// - precomputed-kernel support-vector rows starting with
///   `0:sample_serial_number`.
///
/// This loader does not prove model provenance, semantic correctness relative
/// to a training set, or suitability for a deployment. Malformed text input
/// within the configured caps is returned as [`SvmError`] rather than panicking.
///
/// ### Complexity
///
/// The header parse is `O(nr_class)` in the worst case (due to `rho` /
/// `label` / `nr_sv` array reads). The SV section is linear in the file
/// size. Downstream consumers of the returned [`SvmModel`] — notably
/// `group_classes` and probability estimation — are `O(k²)` on `k =
/// nr_class`, bounded by [`LoadOptions::max_nr_class`].
pub fn load_model(path: &Path) -> Result<SvmModel, SvmError> {
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    load_model_from_reader(reader)
}

/// Load an SVM model from any buffered reader.
///
/// Uses [`LoadOptions::default`]. See [`load_model`] for the validation
/// contract and non-goals. This is the reader-backed equivalent of LIBSVM's
/// `svm_load_model`.
///
/// ```
/// use libsvm_rs::io::{load_model_from_reader, save_model_to_writer};
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
/// let model = svm_train(&problem, &param);
///
/// let mut bytes = Vec::new();
/// save_model_to_writer(&mut bytes, &model)?;
/// let loaded = load_model_from_reader(std::io::Cursor::new(bytes))?;
/// assert_eq!(loaded.sv.len(), model.sv.len());
/// # Ok::<(), libsvm_rs::SvmError>(())
/// ```
pub fn load_model_from_reader(reader: impl BufRead) -> Result<SvmModel, SvmError> {
    load_model_from_reader_with_options(reader, &LoadOptions::default())
}

/// Load an SVM model from any buffered reader, with explicit resource caps.
///
/// See [`LoadOptions`] for the meaning of each cap and for defaults tuned for
/// untrusted input. This function has the same validation contract as
/// [`load_model`], with caller-supplied caps.
pub fn load_model_from_reader_with_options(
    mut reader: impl BufRead,
    options: &LoadOptions,
) -> Result<SvmModel, SvmError> {
    let mut bytes_read: u64 = 0;

    // The `nr_class` / `total_sv` caps are the intersection of the
    // module-level hard limits (`MAX_NR_CLASS`, `MAX_TOTAL_SV`) and the
    // per-call `LoadOptions` overrides. A caller using
    // `LoadOptions::trusted_input()` relaxes only down to the hard caps;
    // it cannot exceed them. This is defense in depth.
    let nr_class_cap = options.max_nr_class.min(MAX_NR_CLASS);
    let total_sv_cap = options.max_sv.min(MAX_TOTAL_SV);

    // Defaults
    let mut param = SvmParameter::default();
    let mut nr_class: usize = 0;
    let mut total_sv: usize = 0;
    let mut rho: Vec<f64> = Vec::new();
    let mut label: Vec<i32> = Vec::new();
    let mut prob_a: Vec<f64> = Vec::new();
    let mut prob_b: Vec<f64> = Vec::new();
    let mut prob_density_marks: Vec<f64> = Vec::new();
    let mut n_sv: Vec<usize> = Vec::new();

    // Read header.
    let mut line_num: usize = 0;
    loop {
        let raw = read_line_capped(
            &mut reader,
            &mut bytes_read,
            options.max_bytes,
            options.max_line_len,
        )
        .map_err(|e| SvmError::ModelFormatError(e.to_string()))?
        .ok_or_else(|| SvmError::ModelFormatError("unexpected end of file in header".into()))?;
        line_num += 1;
        let line = raw.trim().to_string();
        if line.is_empty() {
            continue;
        }

        let mut parts = line.split_whitespace();
        let cmd = parts.next().ok_or_else(|| {
            SvmError::ModelFormatError(format!("line {}: empty model header line", line_num))
        })?;

        match cmd {
            "svm_type" => {
                let val = parts.next().ok_or_else(|| {
                    SvmError::ModelFormatError(format!("line {}: missing svm_type value", line_num))
                })?;
                param.svm_type = str_to_svm_type(val).ok_or_else(|| {
                    SvmError::ModelFormatError(format!(
                        "line {}: unknown svm_type: {}",
                        line_num, val
                    ))
                })?;
            }
            "kernel_type" => {
                let val = parts.next().ok_or_else(|| {
                    SvmError::ModelFormatError(format!(
                        "line {}: missing kernel_type value",
                        line_num
                    ))
                })?;
                param.kernel_type = str_to_kernel_type(val).ok_or_else(|| {
                    SvmError::ModelFormatError(format!(
                        "line {}: unknown kernel_type: {}",
                        line_num, val
                    ))
                })?;
            }
            "degree" => {
                let d: i32 = parse_single(&mut parts, line_num, "degree")?;
                if d < 0 {
                    return Err(SvmError::ModelFormatError(format!(
                        "line {}: degree must be >= 0, got {}",
                        line_num, d
                    )));
                }
                param.degree = d;
            }
            "gamma" => {
                let v: f64 = parse_single(&mut parts, line_num, "gamma")?;
                if !v.is_finite() {
                    return Err(SvmError::ModelFormatError(format!(
                        "line {}: gamma must be finite, got {}",
                        line_num, v
                    )));
                }
                param.gamma = v;
            }
            "coef0" => {
                let v: f64 = parse_single(&mut parts, line_num, "coef0")?;
                if !v.is_finite() {
                    return Err(SvmError::ModelFormatError(format!(
                        "line {}: coef0 must be finite, got {}",
                        line_num, v
                    )));
                }
                param.coef0 = v;
            }
            "nr_class" => {
                nr_class = parse_single(&mut parts, line_num, "nr_class")?;
                if nr_class > nr_class_cap {
                    return Err(SvmError::ModelFormatError(format!(
                        "line {}: nr_class exceeds limit ({})",
                        line_num, nr_class_cap
                    )));
                }
            }
            "total_sv" => {
                total_sv = parse_single(&mut parts, line_num, "total_sv")?;
                if total_sv > total_sv_cap {
                    return Err(SvmError::ModelFormatError(format!(
                        "line {}: total_sv exceeds limit ({})",
                        line_num, total_sv_cap
                    )));
                }
            }
            "rho" => {
                rho = parse_multiple(&mut parts, line_num, "rho")?;
                for &r in &rho {
                    if !r.is_finite() {
                        return Err(SvmError::ModelFormatError(format!(
                            "line {}: rho must be finite, got {}",
                            line_num, r
                        )));
                    }
                }
            }
            "label" => {
                label = parse_multiple(&mut parts, line_num, "label")?;
            }
            "probA" => {
                prob_a = parse_multiple(&mut parts, line_num, "probA")?;
            }
            "probB" => {
                prob_b = parse_multiple(&mut parts, line_num, "probB")?;
            }
            "prob_density_marks" => {
                prob_density_marks = parse_multiple(&mut parts, line_num, "prob_density_marks")?;
            }
            "nr_sv" => {
                n_sv = parts
                    .map(|s| {
                        s.parse::<usize>().map_err(|_| {
                            SvmError::ModelFormatError(format!(
                                "line {}: invalid nr_sv value: {}",
                                line_num, s
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
            }
            "SV" => break,
            _ => {
                return Err(SvmError::ModelFormatError(format!(
                    "line {}: unknown keyword: {}",
                    line_num, cmd
                )));
            }
        }
    }

    // Cross-consistency checks on the header.
    //
    // These run before any per-SV allocation so malformed files are rejected
    // early, and so downstream code can rely on structural invariants (e.g.
    // `rho.len() == nr_class * (nr_class - 1) / 2` for multiclass).
    validate_model_header(
        param.svm_type,
        nr_class,
        total_sv,
        &rho,
        &label,
        &prob_a,
        &prob_b,
        &prob_density_marks,
        &n_sv,
    )?;

    // Read SV section.
    //
    // SECURITY: we do NOT preallocate with `total_sv` capacity. A malicious
    // file could claim up to `MAX_TOTAL_SV` support vectors in its header,
    // which would trigger terabyte-scale reservations before any real data is
    // read. Amortized `Vec` growth caps peak memory at the actually-parsed
    // payload.
    let m = if nr_class > 1 { nr_class - 1 } else { 1 };
    let mut sv_coef: Vec<Vec<f64>> = (0..m).map(|_| Vec::new()).collect();
    let mut sv: Vec<Vec<SvmNode>> = Vec::new();

    while sv.len() < total_sv {
        let raw = read_line_capped(
            &mut reader,
            &mut bytes_read,
            options.max_bytes,
            options.max_line_len,
        )
        .map_err(|e| SvmError::ModelFormatError(e.to_string()))?
        .ok_or_else(|| SvmError::ModelFormatError("unexpected end of file in SV section".into()))?;
        line_num += 1;
        let line = raw.trim();
        if line.is_empty() {
            // Skip blank lines without consuming an SV slot — the header's
            // `total_sv` must match the number of SV rows we actually collect.
            continue;
        }

        let mut parts = line.split_whitespace();

        // First m tokens are sv_coef values
        for (k, coef_row) in sv_coef.iter_mut().enumerate() {
            let val_str = parts.next().ok_or_else(|| {
                SvmError::ModelFormatError(format!("line {}: missing sv_coef[{}]", line_num, k))
            })?;
            let val: f64 = val_str.parse().map_err(|_| {
                SvmError::ModelFormatError(format!(
                    "line {}: invalid sv_coef: {}",
                    line_num, val_str
                ))
            })?;
            if !val.is_finite() {
                return Err(SvmError::ModelFormatError(format!(
                    "line {}: sv_coef must be finite, got {}",
                    line_num, val_str
                )));
            }
            coef_row.push(val);
        }

        // Remaining tokens are index:value pairs (ascending index order, same
        // invariant as the problem-file parser).
        let mut nodes = Vec::new();
        let mut prev_index: i32 = 0;
        for token in parts {
            let (idx_str, val_str) = token.split_once(':').ok_or_else(|| {
                SvmError::ModelFormatError(format!(
                    "line {}: expected index:value, got: {}",
                    line_num, token
                ))
            })?;
            let index: i32 =
                parse_feature_index_model_line(line_num, idx_str, options.max_feature_index)?;

            if !nodes.is_empty() && index <= prev_index {
                return Err(SvmError::ModelFormatError(format!(
                    "line {}: feature indices must be ascending: {} follows {}",
                    line_num, index, prev_index
                )));
            }

            let value: f64 = val_str.parse().map_err(|_| {
                SvmError::ModelFormatError(format!("line {}: invalid value: {}", line_num, val_str))
            })?;
            if !value.is_finite() {
                return Err(SvmError::ModelFormatError(format!(
                    "line {}: feature value must be finite, got {}",
                    line_num, val_str
                )));
            }
            prev_index = index;
            nodes.push(SvmNode { index, value });
        }

        if param.kernel_type == KernelType::Precomputed {
            validate_precomputed_row(&nodes, line_num, "support vector")?;
        }
        sv.push(nodes);
    }

    let model = SvmModel {
        param,
        nr_class,
        sv,
        sv_coef,
        rho,
        prob_a,
        prob_b,
        prob_density_marks,
        sv_indices: Vec::new(), // not stored in model file
        label,
        n_sv,
    };
    validate_model(&model)?;
    Ok(model)
}

/// Validate the shared structural invariants required of a loaded/deserialized model.
pub(crate) fn validate_model(model: &SvmModel) -> Result<(), SvmError> {
    if model.param.degree < 0 {
        return Err(SvmError::ModelFormatError(format!(
            "degree must be >= 0, got {}",
            model.param.degree
        )));
    }
    if !model.param.gamma.is_finite() {
        return Err(SvmError::ModelFormatError(format!(
            "gamma must be finite, got {}",
            model.param.gamma
        )));
    }
    if matches!(
        model.param.kernel_type,
        KernelType::Polynomial | KernelType::Rbf | KernelType::Sigmoid
    ) && model.param.gamma < 0.0
    {
        return Err(SvmError::ModelFormatError(format!(
            "gamma must be >= 0, got {}",
            model.param.gamma
        )));
    }
    if !model.param.coef0.is_finite() {
        return Err(SvmError::ModelFormatError(format!(
            "coef0 must be finite, got {}",
            model.param.coef0
        )));
    }

    validate_model_header(
        model.param.svm_type,
        model.nr_class,
        model.sv.len(),
        &model.rho,
        &model.label,
        &model.prob_a,
        &model.prob_b,
        &model.prob_density_marks,
        &model.n_sv,
    )?;

    let expected_rows = model.nr_class.saturating_sub(1).max(1);
    if model.sv_coef.len() != expected_rows {
        return Err(SvmError::ModelFormatError(format!(
            "sv_coef has {} rows, expected {}",
            model.sv_coef.len(),
            expected_rows
        )));
    }
    for (row_idx, row) in model.sv_coef.iter().enumerate() {
        if row.len() != model.sv.len() {
            return Err(SvmError::ModelFormatError(format!(
                "sv_coef row {} has {} entries, expected {}",
                row_idx,
                row.len(),
                model.sv.len()
            )));
        }
        for &coef in row {
            if !coef.is_finite() {
                return Err(SvmError::ModelFormatError(format!(
                    "sv_coef must be finite, got {}",
                    coef
                )));
            }
        }
    }

    for &r in &model.rho {
        if !r.is_finite() {
            return Err(SvmError::ModelFormatError(format!(
                "rho must be finite, got {}",
                r
            )));
        }
    }
    for (name, values) in [("probA", &model.prob_a), ("probB", &model.prob_b)] {
        for &value in values {
            if !value.is_finite() {
                return Err(SvmError::ModelFormatError(format!(
                    "{} must be finite, got {}",
                    name, value
                )));
            }
        }
    }
    for &value in &model.prob_density_marks {
        if !value.is_finite() {
            return Err(SvmError::ModelFormatError(format!(
                "prob_density_marks must be finite, got {}",
                value
            )));
        }
    }

    for (row_idx, nodes) in model.sv.iter().enumerate() {
        let mut prev_index = 0;
        for (node_idx, node) in nodes.iter().enumerate() {
            if node_idx > 0 && node.index <= prev_index {
                return Err(SvmError::ModelFormatError(format!(
                    "support vector {} feature indices must be ascending: {} follows {}",
                    row_idx + 1,
                    node.index,
                    prev_index
                )));
            }
            if !node.value.is_finite() {
                return Err(SvmError::ModelFormatError(format!(
                    "feature value must be finite, got {}",
                    node.value
                )));
            }
            prev_index = node.index;
        }
        if model.param.kernel_type == KernelType::Precomputed {
            validate_precomputed_row(nodes, row_idx + 1, "support vector")?;
        }
    }

    Ok(())
}

fn validate_precomputed_row(
    nodes: &[SvmNode],
    line_num: usize,
    context: &str,
) -> Result<(), SvmError> {
    let first = nodes.first().ok_or_else(|| {
        SvmError::ModelFormatError(format!(
            "line {}: precomputed kernel {} is missing 0:sample_serial_number",
            line_num, context
        ))
    })?;

    if first.index != 0
        || !first.value.is_finite()
        || first.value < 1.0
        || first.value.fract() != 0.0
    {
        return Err(SvmError::ModelFormatError(format!(
            "line {}: precomputed kernel {} must start with 0:sample_serial_number",
            line_num, context
        )));
    }

    Ok(())
}

// ─── Cross-consistency validation ────────────────────────────────────

/// Validate model-header invariants before reading the SV section.
///
/// A malformed or adversarial model file can pass individual field parses
/// and still describe a structurally impossible SVM. This gate rejects the
/// mismatch early, before any allocation keyed on `total_sv`, so downstream
/// code (prediction, probability estimation) can rely on the usual LIBSVM
/// shape contracts:
///
/// * `rho.len() == k * (k - 1) / 2` for `k = nr_class` on classification,
///   `rho.len() == 1` for one-class / regression (where `nr_class == 2`).
/// * `label.len() == nr_class` and `n_sv.len() == nr_class` if supplied.
/// * `sum(n_sv) == total_sv` if `n_sv` is supplied.
/// * `prob_a` / `prob_b` (if supplied) match the expected decision-function
///   count, and `prob_density_marks` only appears on one-class models.
///
/// Optional fields (e.g. `label`, `n_sv`, `probA`) are only validated when
/// present, because minimal hand-written fixtures and some legacy writers
/// omit them; the invariant "if present, must be consistent" is what matters
/// for safety.
#[allow(clippy::too_many_arguments)]
fn validate_model_header(
    svm_type: SvmType,
    nr_class: usize,
    total_sv: usize,
    rho: &[f64],
    label: &[i32],
    prob_a: &[f64],
    prob_b: &[f64],
    prob_density_marks: &[f64],
    n_sv: &[usize],
) -> Result<(), SvmError> {
    let is_classification = matches!(svm_type, SvmType::CSvc | SvmType::NuSvc);
    let is_regression = matches!(svm_type, SvmType::EpsilonSvr | SvmType::NuSvr);
    let is_one_class = matches!(svm_type, SvmType::OneClass);

    // nr_class must be at least 2 under the LIBSVM convention (regression and
    // one-class store nr_class=2 as well, because the one-vs-one scaffolding
    // is reused). nr_class==0 or 1 would yield `m = nr_class - 1 = 0` or
    // underflow-prone arithmetic elsewhere.
    if nr_class < 2 {
        return Err(SvmError::ModelFormatError(format!(
            "nr_class must be >= 2, got {}",
            nr_class
        )));
    }

    // Expected rho length depends on svm_type.
    let expected_rho = if is_classification {
        nr_class * (nr_class - 1) / 2
    } else {
        1
    };
    if rho.len() != expected_rho {
        return Err(SvmError::ModelFormatError(format!(
            "rho has {} entries, expected {} for svm_type {}",
            rho.len(),
            expected_rho,
            svm_type_to_str(svm_type)
        )));
    }

    // label is mandatory for classification types; must equal nr_class when present.
    // For one_class / epsilon_svr / nu_svr, label is absent and empty is legitimate.
    if is_classification {
        // Require label to be present and correctly sized; an empty label vec on a
        // classification model means the file omitted the `label` line entirely, which
        // would leave predict.rs indexing `model.label[i]` out of bounds.
        if label.len() != nr_class {
            return Err(SvmError::ModelFormatError(format!(
                "label has {} entries, expected nr_class ({}) for svm_type {}",
                label.len(),
                nr_class,
                svm_type_to_str(svm_type)
            )));
        }
    } else if !label.is_empty() {
        return Err(SvmError::ModelFormatError(format!(
            "label is only valid for classification, got {} entries on svm_type {}",
            label.len(),
            svm_type_to_str(svm_type)
        )));
    }

    // n_sv: mandatory for classification; absent on regression/one-class.
    // An empty n_sv on a classification model means the file omitted `nr_sv`,
    // which would leave predict.rs indexing `model.n_sv[i-1]` out of bounds.
    if is_classification {
        if n_sv.len() != nr_class {
            return Err(SvmError::ModelFormatError(format!(
                "nr_sv has {} entries, expected nr_class ({}) for svm_type {}",
                n_sv.len(),
                nr_class,
                svm_type_to_str(svm_type)
            )));
        }
        // Use checked_add to prevent silent overflow on malicious huge values.
        // MAX_TOTAL_SV bounds total_sv already; n_sv values are parsed as
        // `usize` and otherwise unbounded until this sum-check.
        let mut sum: usize = 0;
        for &n in n_sv {
            sum = sum.checked_add(n).ok_or_else(|| {
                SvmError::ModelFormatError("nr_sv entries overflow usize when summed".into())
            })?;
        }
        if sum != total_sv {
            return Err(SvmError::ModelFormatError(format!(
                "sum of nr_sv entries ({}) does not match total_sv ({})",
                sum, total_sv
            )));
        }
    } else if !n_sv.is_empty() {
        return Err(SvmError::ModelFormatError(format!(
            "nr_sv is only valid for classification, got {} entries on svm_type {}",
            n_sv.len(),
            svm_type_to_str(svm_type)
        )));
    }

    // Probability arrays: must either be absent or length-match rho.
    if !prob_a.is_empty() && prob_a.len() != expected_rho {
        return Err(SvmError::ModelFormatError(format!(
            "probA has {} entries, expected {}",
            prob_a.len(),
            expected_rho
        )));
    }
    if !prob_b.is_empty() && prob_b.len() != expected_rho {
        return Err(SvmError::ModelFormatError(format!(
            "probB has {} entries, expected {}",
            prob_b.len(),
            expected_rho
        )));
    }

    // prob_density_marks is only meaningful for one-class.
    if !prob_density_marks.is_empty() && !is_one_class {
        return Err(SvmError::ModelFormatError(format!(
            "prob_density_marks is only valid for one-class SVM, got {} entries on svm_type {}",
            prob_density_marks.len(),
            svm_type_to_str(svm_type)
        )));
    }

    // Regression/one-class should not carry classification-only artifacts.
    // (Already caught above via `label` / `n_sv` branches; this assertion
    // keeps the intent self-documenting for future maintainers.)
    let _ = is_regression;

    Ok(())
}

// ─── Helper parsers ──────────────────────────────────────────────────

fn parse_feature_index_problem_line(
    line_num: usize,
    idx_str: &str,
    max_feature_index: i32,
) -> Result<i32, SvmError> {
    parse_feature_index(idx_str, max_feature_index).map_err(|msg| SvmError::ParseError {
        line: line_num,
        message: msg,
    })
}

fn parse_feature_index_model_line(
    line_num: usize,
    idx_str: &str,
    max_feature_index: i32,
) -> Result<i32, SvmError> {
    parse_feature_index(idx_str, max_feature_index)
        .map_err(|msg| SvmError::ModelFormatError(format!("line {}: {}", line_num, msg)))
}

fn parse_single<T: std::str::FromStr>(
    parts: &mut std::str::SplitWhitespace<'_>,
    line_num: usize,
    field: &str,
) -> Result<T, SvmError> {
    let val_str = parts.next().ok_or_else(|| {
        SvmError::ModelFormatError(format!("line {}: missing {} value", line_num, field))
    })?;
    val_str.parse().map_err(|_| {
        SvmError::ModelFormatError(format!(
            "line {}: invalid {} value: {}",
            line_num, field, val_str
        ))
    })
}

fn parse_multiple<T: std::str::FromStr>(
    parts: &mut std::str::SplitWhitespace<'_>,
    line_num: usize,
    field: &str,
) -> Result<Vec<T>, SvmError> {
    parts
        .map(|s| {
            s.parse::<T>().map_err(|_| {
                SvmError::ModelFormatError(format!(
                    "line {}: invalid {} value: {}",
                    line_num, field, s
                ))
            })
        })
        .collect()
}

// ─── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn data_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("data")
    }

    #[test]
    fn parse_heart_scale() {
        let path = data_dir().join("heart_scale");
        let problem = load_problem(&path).unwrap();
        assert_eq!(problem.labels.len(), 270);
        assert_eq!(problem.instances.len(), 270);
        // First instance: +1 label, 12 features (index 11 is missing/sparse)
        assert_eq!(problem.labels[0], 1.0);
        assert_eq!(
            problem.instances[0][0],
            SvmNode {
                index: 1,
                value: 0.708333
            }
        );
        assert_eq!(problem.instances[0].len(), 12);
    }

    #[test]
    fn parse_iris() {
        let path = data_dir().join("iris.scale");
        let problem = load_problem(&path).unwrap();
        assert_eq!(problem.labels.len(), 150);
        // 3 classes: 1, 2, 3
        let classes: std::collections::HashSet<i64> =
            problem.labels.iter().map(|&l| l as i64).collect();
        assert_eq!(classes.len(), 3);
    }

    #[test]
    fn parse_housing() {
        let path = data_dir().join("housing_scale");
        let problem = load_problem(&path).unwrap();
        assert_eq!(problem.labels.len(), 506);
        // Regression: labels are continuous
        assert!((problem.labels[0] - 24.0).abs() < 1e-10);
    }

    #[test]
    fn parse_empty_lines() {
        let input = b"+1 1:0.5\n\n-1 2:0.3\n";
        let problem = load_problem_from_reader(&input[..]).unwrap();
        assert_eq!(problem.labels.len(), 2);
    }

    #[test]
    fn parse_error_unsorted_indices() {
        let input = b"+1 3:0.5 1:0.3\n";
        let result = load_problem_from_reader(&input[..]);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("ascending"), "error: {}", msg);
    }

    #[test]
    fn parse_error_duplicate_indices() {
        let input = b"+1 1:0.5 1:0.3\n";
        let result = load_problem_from_reader(&input[..]);
        assert!(result.is_err());
    }

    #[test]
    fn parse_error_missing_colon() {
        let input = b"+1 1:0.5 bad_token\n";
        let result = load_problem_from_reader(&input[..]);
        assert!(result.is_err());
    }

    #[test]
    #[allow(clippy::excessive_precision)]
    fn load_c_trained_model() {
        // Load a model produced by the original C LIBSVM svm-train
        let path = data_dir().join("heart_scale.model");
        let model = load_model(&path).unwrap();
        assert_eq!(model.nr_class, 2);
        assert_eq!(model.param.svm_type, SvmType::CSvc);
        assert_eq!(model.param.kernel_type, KernelType::Rbf);
        assert!((model.param.gamma - 0.076923076923076927).abs() < 1e-15);
        assert_eq!(model.sv.len(), 132);
        assert_eq!(model.label, vec![1, -1]);
        assert_eq!(model.n_sv, vec![64, 68]);
        assert!((model.rho[0] - 0.42446205176771573).abs() < 1e-15);
        // sv_coef should have 1 row (nr_class - 1) with 132 entries
        assert_eq!(model.sv_coef.len(), 1);
        assert_eq!(model.sv_coef[0].len(), 132);
    }

    #[test]
    fn roundtrip_c_model() {
        // Load C model, save it back, and verify byte-exact match
        let path = data_dir().join("heart_scale.model");
        let original_bytes = std::fs::read_to_string(&path).unwrap();
        let model = load_model(&path).unwrap();

        let mut buf = Vec::new();
        save_model_to_writer(&mut buf, &model).unwrap();
        let rust_output = String::from_utf8(buf).unwrap();

        // Compare line by line for better diagnostics
        let orig_lines: Vec<&str> = original_bytes.lines().collect();
        let rust_lines: Vec<&str> = rust_output.lines().collect();
        assert_eq!(
            orig_lines.len(),
            rust_lines.len(),
            "line count mismatch: C={} Rust={}",
            orig_lines.len(),
            rust_lines.len()
        );
        for (i, (o, r)) in orig_lines.iter().zip(rust_lines.iter()).enumerate() {
            assert_eq!(
                o,
                r,
                "line {} differs:\n  C:    {:?}\n  Rust: {:?}",
                i + 1,
                o,
                r
            );
        }
    }

    #[test]
    #[allow(clippy::excessive_precision)]
    fn gfmt_matches_c_printf() {
        // Reference values from C's printf("%.17g|%.8g\n", v, v)
        let cases: &[(f64, &str, &str)] = &[
            (0.5, "0.5", "0.5"),
            (-1.0, "-1", "-1"),
            (0.123456789012345, "0.123456789012345", "0.12345679"),
            (-0.987654321098765, "-0.98765432109876505", "-0.98765432"),
            (0.42446200000000001, "0.42446200000000001", "0.424462"),
            (0.0, "0", "0"),
            (1e-5, "1.0000000000000001e-05", "1e-05"),
            (1e-4, "0.0001", "0.0001"),
            (1e20, "1e+20", "1e+20"),
            (-0.25, "-0.25", "-0.25"),
            (0.75, "0.75", "0.75"),
            (0.708333, "0.70833299999999999", "0.708333"),
            (1.0, "1", "1"),
        ];
        for &(v, expected_17g, expected_8g) in cases {
            let got_17 = format!("{}", fmt_17g(v));
            let got_8 = format!("{}", fmt_8g(v));
            assert_eq!(got_17, expected_17g, "%.17g mismatch for {}", v);
            assert_eq!(got_8, expected_8g, "%.8g mismatch for {}", v);
        }
    }

    #[test]
    #[allow(clippy::excessive_precision)]
    fn model_roundtrip() {
        // Create a minimal model and verify save → load roundtrip
        let model = SvmModel {
            param: SvmParameter {
                svm_type: SvmType::CSvc,
                kernel_type: KernelType::Rbf,
                gamma: 0.5,
                ..Default::default()
            },
            nr_class: 2,
            sv: vec![
                vec![
                    SvmNode {
                        index: 1,
                        value: 0.5,
                    },
                    SvmNode {
                        index: 3,
                        value: -1.0,
                    },
                ],
                vec![
                    SvmNode {
                        index: 1,
                        value: -0.25,
                    },
                    SvmNode {
                        index: 2,
                        value: 0.75,
                    },
                ],
            ],
            sv_coef: vec![vec![0.123456789012345, -0.987654321098765]],
            rho: vec![0.42446200000000001],
            prob_a: vec![],
            prob_b: vec![],
            prob_density_marks: vec![],
            sv_indices: vec![],
            label: vec![1, -1],
            n_sv: vec![1, 1],
        };

        let mut buf = Vec::new();
        save_model_to_writer(&mut buf, &model).unwrap();

        let loaded = load_model_from_reader(&buf[..]).unwrap();

        assert_eq!(loaded.nr_class, model.nr_class);
        assert_eq!(loaded.param.svm_type, model.param.svm_type);
        assert_eq!(loaded.param.kernel_type, model.param.kernel_type);
        assert_eq!(loaded.sv.len(), model.sv.len());
        assert_eq!(loaded.label, model.label);
        assert_eq!(loaded.n_sv, model.n_sv);
        assert_eq!(loaded.rho.len(), model.rho.len());
        // Check rho within tolerance (roundtrip through text)
        for (a, b) in loaded.rho.iter().zip(model.rho.iter()) {
            assert!((a - b).abs() < 1e-10, "rho mismatch: {} vs {}", a, b);
        }
        // Check sv_coef within tolerance
        for (row_a, row_b) in loaded.sv_coef.iter().zip(model.sv_coef.iter()) {
            for (a, b) in row_a.iter().zip(row_b.iter()) {
                assert!((a - b).abs() < 1e-10, "sv_coef mismatch: {} vs {}", a, b);
            }
        }
    }

    #[test]
    fn parse_error_excessive_counts() {
        let input =
            b"svm_type c_svc\nkernel_type linear\nnr_class 1000000\ntotal_sv 100\nrho 0\nSV\n";
        let result = load_model_from_reader(&input[..]);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("nr_class exceeds limit"));

        let input =
            b"svm_type c_svc\nkernel_type linear\nnr_class 2\ntotal_sv 100000000\nrho 0\nSV\n";
        let result = load_model_from_reader(&input[..]);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("total_sv exceeds limit"));
    }

    #[test]
    fn parse_error_excessive_feature_index() {
        // Problem file
        let input = b"1 10000001:1\n";
        let result = load_problem_from_reader(&input[..]);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("feature index 10000001 exceeds limit"));

        // Model file
        let input = b"svm_type c_svc\nkernel_type linear\nnr_class 2\ntotal_sv 1\nrho 0\nlabel 1 -1\nnr_sv 1 0\nSV\n0.1 10000001:1\n";
        let result = load_model_from_reader(&input[..]);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("feature index 10000001 exceeds limit"));
    }

    #[test]
    fn parse_error_unknown_model_keyword() {
        let input = b"bad_key value\n";
        let result = load_model_from_reader(&input[..]);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("unknown keyword"));
    }

    #[test]
    fn parse_error_missing_or_unknown_model_values() {
        let missing = b"svm_type\n";
        let err = load_model_from_reader(&missing[..]).unwrap_err();
        assert!(format!("{}", err).contains("missing svm_type value"));

        let unknown = b"svm_type unknown_type\n";
        let err = load_model_from_reader(&unknown[..]).unwrap_err();
        assert!(format!("{}", err).contains("unknown svm_type"));
    }

    #[test]
    fn parse_error_invalid_nr_sv_entry() {
        let input = b"svm_type c_svc\n\
kernel_type linear\n\
nr_class 2\n\
total_sv 1\n\
rho 0\n\
label 1 -1\n\
nr_sv a 1\n\
SV\n\
0.1 1:0.5\n";
        let err = load_model_from_reader(&input[..]).unwrap_err();
        assert!(format!("{}", err).contains("invalid nr_sv value"));
    }

    #[test]
    fn parse_error_in_sv_section_tokens() {
        let missing_coef = b"svm_type c_svc\n\
kernel_type linear\n\
nr_class 2\n\
total_sv 1\n\
rho 0\n\
label 1 -1\n\
nr_sv 1 0\n\
SV\n\
1:0.5\n";
        let err = load_model_from_reader(&missing_coef[..]).unwrap_err();
        assert!(format!("{}", err).contains("invalid sv_coef"));

        let bad_feature = b"svm_type c_svc\n\
kernel_type linear\n\
nr_class 2\n\
total_sv 1\n\
rho 0\n\
label 1 -1\n\
nr_sv 1 0\n\
SV\n\
0.1 bad\n";
        let err = load_model_from_reader(&bad_feature[..]).unwrap_err();
        assert!(format!("{}", err).contains("expected index:value"));
    }

    #[test]
    fn parse_error_unexpected_eof_in_header_and_sv_section() {
        let eof_header = b"svm_type c_svc\n";
        let err = load_model_from_reader(&eof_header[..]).unwrap_err();
        assert!(format!("{}", err).contains("unexpected end of file in header"));

        let eof_sv = b"svm_type c_svc\n\
kernel_type linear\n\
nr_class 2\n\
total_sv 2\n\
rho 0\n\
label 1 -1\n\
nr_sv 1 1\n\
SV\n\
0.1 1:0.5\n";
        let err = load_model_from_reader(&eof_sv[..]).unwrap_err();
        assert!(format!("{}", err).contains("unexpected end of file in SV section"));
    }

    #[test]
    fn reject_rho_length_mismatch_for_classification() {
        // CSvc with nr_class=3 expects rho.len() == 3. Supplying 1 entry
        // reveals either a malformed file or an intentional substitution.
        let input = b"svm_type c_svc\n\
kernel_type linear\n\
nr_class 3\n\
total_sv 3\n\
rho 0\n\
label 1 -1 0\n\
nr_sv 1 1 1\n\
SV\n";
        let err = load_model_from_reader(&input[..]).unwrap_err();
        assert!(
            format!("{}", err).contains("rho has 1 entries, expected 3"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn reject_rho_length_mismatch_for_regression() {
        // SVR expects exactly 1 rho entry; 2 entries is inconsistent.
        let input = b"svm_type epsilon_svr\n\
kernel_type linear\n\
nr_class 2\n\
total_sv 0\n\
rho 0 1\n\
SV\n";
        let err = load_model_from_reader(&input[..]).unwrap_err();
        assert!(
            format!("{}", err).contains("rho has 2 entries, expected 1"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn reject_label_on_regression() {
        // label is classification-only; carrying it on SVR is a red flag.
        let input = b"svm_type epsilon_svr\n\
kernel_type linear\n\
nr_class 2\n\
total_sv 0\n\
rho 0\n\
label 1 -1\n\
SV\n";
        let err = load_model_from_reader(&input[..]).unwrap_err();
        assert!(
            format!("{}", err).contains("label is only valid for classification"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn reject_label_length_mismatch() {
        let input = b"svm_type c_svc\n\
kernel_type linear\n\
nr_class 3\n\
total_sv 0\n\
rho 0 0 0\n\
label 1 -1\n\
SV\n";
        let err = load_model_from_reader(&input[..]).unwrap_err();
        assert!(
            format!("{}", err).contains("label has 2 entries, expected nr_class (3)"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn reject_nr_sv_sum_mismatch() {
        // total_sv=5 but nr_sv sums to 3. An inconsistent header like this
        // could previously pass and leave downstream code with stale assumptions.
        let input = b"svm_type c_svc\n\
kernel_type linear\n\
nr_class 2\n\
total_sv 5\n\
rho 0\n\
label 1 -1\n\
nr_sv 1 2\n\
SV\n";
        let err = load_model_from_reader(&input[..]).unwrap_err();
        assert!(
            format!("{}", err).contains("sum of nr_sv entries (3) does not match total_sv (5)"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn reject_nr_sv_length_mismatch() {
        let input = b"svm_type c_svc\n\
kernel_type linear\n\
nr_class 3\n\
total_sv 3\n\
rho 0 0 0\n\
label 1 -1 0\n\
nr_sv 1 2\n\
SV\n";
        let err = load_model_from_reader(&input[..]).unwrap_err();
        assert!(
            format!("{}", err).contains("nr_sv has 2 entries, expected nr_class (3)"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn reject_proba_length_mismatch() {
        let input = b"svm_type c_svc\n\
kernel_type linear\n\
nr_class 3\n\
total_sv 0\n\
rho 0 0 0\n\
label 1 -1 0\n\
nr_sv 0 0 0\n\
probA 0.1 0.2\n\
SV\n";
        let err = load_model_from_reader(&input[..]).unwrap_err();
        assert!(
            format!("{}", err).contains("probA has 2 entries, expected 3"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn reject_prob_density_marks_on_csvc() {
        let input = b"svm_type c_svc\n\
kernel_type linear\n\
nr_class 2\n\
total_sv 0\n\
rho 0\n\
label 1 -1\n\
nr_sv 0 0\n\
prob_density_marks 0.1 0.2\n\
SV\n";
        let err = load_model_from_reader(&input[..]).unwrap_err();
        assert!(
            format!("{}", err).contains("prob_density_marks is only valid for one-class SVM"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn reject_nr_class_below_two() {
        let input = b"svm_type c_svc\n\
kernel_type linear\n\
nr_class 1\n\
total_sv 0\n\
rho\n\
SV\n";
        let err = load_model_from_reader(&input[..]).unwrap_err();
        assert!(
            format!("{}", err).contains("nr_class must be >= 2, got 1"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn reject_sv_feature_indices_not_ascending() {
        let input = b"svm_type c_svc\n\
kernel_type linear\n\
nr_class 2\n\
total_sv 1\n\
rho 0\n\
label 1 -1\n\
nr_sv 1 0\n\
SV\n\
0.1 3:0.5 1:0.3\n";
        let err = load_model_from_reader(&input[..]).unwrap_err();
        assert!(
            format!("{}", err).contains("feature indices must be ascending"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn reject_precomputed_model_sv_without_sample_serial_number() {
        let input = b"svm_type c_svc\n\
kernel_type precomputed\n\
nr_class 2\n\
total_sv 1\n\
rho 0\n\
label 1 -1\n\
nr_sv 1 0\n\
SV\n\
0.1\n";
        let err = load_model_from_reader(&input[..]).unwrap_err();
        assert!(
            format!("{}", err).contains("missing 0:sample_serial_number"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn reject_sv_feature_index_duplicated() {
        let input = b"svm_type c_svc\n\
kernel_type linear\n\
nr_class 2\n\
total_sv 1\n\
rho 0\n\
label 1 -1\n\
nr_sv 1 0\n\
SV\n\
0.1 1:0.5 1:0.3\n";
        let err = load_model_from_reader(&input[..]).unwrap_err();
        assert!(
            format!("{}", err).contains("feature indices must be ascending"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn load_options_default_caps_match_documented_values() {
        let opts = LoadOptions::default();
        assert_eq!(opts.max_bytes, 64 * 1024 * 1024);
        assert_eq!(opts.max_line_len, 1024 * 1024);
        assert_eq!(opts.max_sv, MAX_TOTAL_SV);
        assert_eq!(opts.max_nr_class, MAX_NR_CLASS);
        assert_eq!(opts.max_feature_index, MAX_FEATURE_INDEX);
    }

    #[test]
    fn load_options_trusted_input_sets_type_maxes() {
        let opts = LoadOptions::trusted_input();
        assert_eq!(opts.max_bytes, u64::MAX);
        assert_eq!(opts.max_line_len, usize::MAX);
        assert_eq!(opts.max_sv, usize::MAX);
        assert_eq!(opts.max_nr_class, usize::MAX);
        assert_eq!(opts.max_feature_index, i32::MAX);
    }

    #[test]
    fn problem_reader_rejects_file_over_max_bytes() {
        // 20 bytes of content, cap at 10.
        let input = b"+1 1:0.5\n+1 2:0.5\n";
        let opts = LoadOptions {
            max_bytes: 10,
            ..LoadOptions::default()
        };
        let err = load_problem_from_reader_with_options(&input[..], &opts).unwrap_err();
        assert!(
            format!("{}", err).contains("max_bytes"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn problem_reader_rejects_line_over_max_line_len() {
        // Build a line of 200 chars; cap max_line_len at 50.
        let mut payload = String::from("+1 ");
        for i in 1..=50 {
            payload.push_str(&format!("{}:0.1 ", i));
        }
        payload.push('\n');
        let opts = LoadOptions {
            max_line_len: 50,
            ..LoadOptions::default()
        };
        let err = load_problem_from_reader_with_options(payload.as_bytes(), &opts).unwrap_err();
        assert!(
            format!("{}", err).contains("max_line_len"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn problem_reader_accepts_line_at_max_line_len() {
        // A line whose content (excluding trailing newline) is exactly
        // max_line_len bytes must be accepted — the cap is inclusive.
        let line_content = "+1 1:0.5";
        let payload = format!("{}\n", line_content);
        let opts = LoadOptions {
            max_line_len: line_content.len(),
            ..LoadOptions::default()
        };
        let problem = load_problem_from_reader_with_options(payload.as_bytes(), &opts).unwrap();
        assert_eq!(problem.labels.len(), 1);
    }

    #[test]
    fn problem_reader_tolerates_crlf_at_cap() {
        // Same content length, but with \r\n. Total bytes on disk is
        // content_len + 2; the helper allows one byte of slack for the \r.
        let line_content = "+1 1:0.5";
        let payload = format!("{}\r\n", line_content);
        let opts = LoadOptions {
            max_line_len: line_content.len(),
            ..LoadOptions::default()
        };
        let problem = load_problem_from_reader_with_options(payload.as_bytes(), &opts).unwrap();
        assert_eq!(problem.labels.len(), 1);
    }

    #[test]
    fn problem_reader_rejects_nul_byte() {
        // A stray NUL byte inside a line is rejected early — LIBSVM text
        // files are ASCII, and a NUL is most likely truncated binary data.
        let mut payload: Vec<u8> = b"+1 1:0.5".to_vec();
        payload.push(0);
        payload.extend_from_slice(b"\n");
        let err = load_problem_from_reader(payload.as_slice()).unwrap_err();
        assert!(
            format!("{}", err).contains("NUL byte"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn model_reader_honors_max_nr_class_cap() {
        // 100 is below both MAX_NR_CLASS and the 50 cap; the 50 cap should
        // fire first.
        let input = b"svm_type c_svc\n\
kernel_type linear\n\
nr_class 100\n\
total_sv 1\n\
rho 0\n\
SV\n";
        let opts = LoadOptions {
            max_nr_class: 50,
            ..LoadOptions::default()
        };
        let err = load_model_from_reader_with_options(&input[..], &opts).unwrap_err();
        assert!(
            format!("{}", err).contains("nr_class exceeds limit (50)"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn model_reader_honors_max_sv_cap() {
        let input = b"svm_type c_svc\n\
kernel_type linear\n\
nr_class 2\n\
total_sv 1000\n\
rho 0\n\
SV\n";
        let opts = LoadOptions {
            max_sv: 100,
            ..LoadOptions::default()
        };
        let err = load_model_from_reader_with_options(&input[..], &opts).unwrap_err();
        assert!(
            format!("{}", err).contains("total_sv exceeds limit (100)"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn trusted_input_cannot_exceed_hard_module_caps() {
        // `LoadOptions::trusted_input()` sets caps to usize::MAX / u64::MAX,
        // but the module-level hard caps (MAX_NR_CLASS, MAX_TOTAL_SV) still
        // apply as an upper bound. This is defense-in-depth.
        let huge_nr_class = format!(
            "svm_type c_svc\n\
kernel_type linear\n\
nr_class {}\n\
total_sv 1\n\
rho 0\n\
SV\n",
            MAX_NR_CLASS + 1
        );
        let opts = LoadOptions::trusted_input();
        let err = load_model_from_reader_with_options(huge_nr_class.as_bytes(), &opts).unwrap_err();
        assert!(
            format!("{}", err).contains("nr_class exceeds limit"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn model_reader_honors_max_feature_index_cap() {
        let input = b"svm_type c_svc\n\
kernel_type linear\n\
nr_class 2\n\
total_sv 1\n\
rho 0\n\
label 1 -1\n\
nr_sv 1 0\n\
SV\n\
0.1 50:0.5\n";
        let opts = LoadOptions {
            max_feature_index: 10,
            ..LoadOptions::default()
        };
        let err = load_model_from_reader_with_options(&input[..], &opts).unwrap_err();
        assert!(
            format!("{}", err).contains("feature index 50 exceeds limit (10)"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn sv_count_loop_counts_nonblank_lines_only() {
        // A blank line inside the SV section must NOT be billed against
        // total_sv — parsing must terminate only after `total_sv` real SV
        // rows have been collected. This guards the loop rewrite from
        // `for _ in 0..total_sv { if empty { continue } ... }`, which would
        // silently accept a model with fewer SVs than the header claims.
        let input = b"svm_type c_svc\n\
kernel_type linear\n\
nr_class 2\n\
total_sv 2\n\
rho 0\n\
label 1 -1\n\
nr_sv 1 1\n\
SV\n\
\n\
0.1 1:0.5\n\
\n\
-0.1 2:0.5\n";
        let model = load_model_from_reader(&input[..]).unwrap();
        assert_eq!(model.sv.len(), 2);
        assert_eq!(model.sv_coef[0].len(), 2);
    }

    #[test]
    fn save_precomputed_model_writes_zero_index() {
        let model = SvmModel {
            param: SvmParameter {
                svm_type: SvmType::CSvc,
                kernel_type: KernelType::Precomputed,
                ..Default::default()
            },
            nr_class: 2,
            sv: vec![vec![SvmNode {
                index: 0,
                value: 7.0,
            }]],
            sv_coef: vec![vec![0.25]],
            rho: vec![0.0],
            prob_a: vec![],
            prob_b: vec![],
            prob_density_marks: vec![],
            sv_indices: vec![],
            label: vec![1, -1],
            n_sv: vec![1, 0],
        };

        let mut buf = Vec::new();
        save_model_to_writer(&mut buf, &model).unwrap();
        let out = String::from_utf8(buf).unwrap();
        assert!(out.contains("kernel_type precomputed"));
        assert!(out.contains("0:7"));
    }

    // ─── F1: classification models must carry label + nr_sv ─────────────

    /// A c_svc model that omits both `label` and `nr_sv` must return a
    /// ParseError rather than panic in predict.rs.
    #[test]
    fn csvc_missing_label_and_nr_sv_returns_error_not_panic() {
        let input = b"svm_type c_svc\n\
kernel_type linear\n\
nr_class 2\n\
total_sv 1\n\
rho 0\n\
SV\n\
1 1:1\n";
        let err = load_model_from_reader(&input[..]).unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("label has 0 entries, expected nr_class (2)"),
            "unexpected error: {}",
            msg
        );
    }

    /// A c_svc model with `label` present but `nr_sv` absent must also fail.
    #[test]
    fn csvc_label_present_nr_sv_absent_returns_error() {
        let input = b"svm_type c_svc\n\
kernel_type linear\n\
nr_class 2\n\
total_sv 1\n\
rho 0\n\
label 1 -1\n\
SV\n\
1 1:1\n";
        let err = load_model_from_reader(&input[..]).unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("nr_sv has 0 entries, expected nr_class (2)"),
            "unexpected error: {}",
            msg
        );
    }

    /// A valid one_class model with no label/nr_sv lines must still load Ok
    /// (regression guard: the new classification-only requirement must not
    /// affect one-class models).
    #[test]
    fn one_class_without_label_and_nr_sv_loads_ok() {
        // one_class uses m = nr_class - 1 = 1 sv_coef column per SV row.
        let input = b"svm_type one_class\n\
kernel_type rbf\n\
gamma 0.5\n\
nr_class 2\n\
total_sv 1\n\
rho -0.5\n\
SV\n\
1 1:0.5\n";
        let model = load_model_from_reader(&input[..]).unwrap();
        assert_eq!(model.sv.len(), 1);
        assert_eq!(model.label, Vec::<i32>::new());
        assert_eq!(model.n_sv, Vec::<usize>::new());
    }

    // ─── F3: non-finite numeric fields must be rejected ──────────────────

    /// A model with `rho inf` must fail with a ParseError.
    #[test]
    fn model_rho_inf_returns_error() {
        let input = b"svm_type c_svc\n\
kernel_type linear\n\
nr_class 2\n\
total_sv 1\n\
rho inf\n\
label 1 -1\n\
nr_sv 1 0\n\
SV\n\
1 1:0.5\n";
        let err = load_model_from_reader(&input[..]).unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("rho must be finite"),
            "unexpected error: {}",
            msg
        );
    }

    /// A model with `1:nan` in an SV feature must fail with a ParseError.
    #[test]
    fn model_sv_feature_nan_returns_error() {
        let input = b"svm_type c_svc\n\
kernel_type linear\n\
nr_class 2\n\
total_sv 1\n\
rho 0\n\
label 1 -1\n\
nr_sv 1 0\n\
SV\n\
1 1:nan\n";
        let err = load_model_from_reader(&input[..]).unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("feature value must be finite"),
            "unexpected error: {}",
            msg
        );
    }
}
