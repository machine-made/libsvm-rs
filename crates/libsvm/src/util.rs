//! Shared internal utilities for libsvm-rs.

#![cfg_attr(
    test,
    expect(
        clippy::unwrap_used,
        reason = "Tests unwrap fixture loading and checked assertions"
    )
)]

use std::sync::{Mutex, OnceLock};

/// Maximum supported feature index.
pub const MAX_FEATURE_INDEX: i32 = 10_000_000;

/// Result of grouping training labels by class.
#[derive(Debug, Clone)]
pub(crate) struct GroupedClasses {
    pub label: Vec<i32>,
    pub start: Vec<usize>,
    pub count: Vec<usize>,
    pub perm: Vec<usize>,
}

/// Group class labels into contiguous blocks.
///
/// Behavior matches LIBSVM's `group_classes`/`svm_group_classes` helpers:
/// - preserves insertion order for first-seen labels,
/// - swaps two-class {-1,1} to {1,-1} canonical ordering,
/// - returns per-class counts, start offsets, and a permutation of rows.
#[expect(
    clippy::cast_possible_truncation,
    reason = "C truncates the same values at these points"
)]
pub(crate) fn group_classes(labels: &[f64]) -> Option<GroupedClasses> {
    let l = labels.len();
    let mut label_list: Vec<i32> = Vec::new();
    let mut count: Vec<usize> = Vec::new();
    let mut data_label = vec![0usize; l];

    for (slot, &label) in data_label.iter_mut().zip(labels) {
        let this_label = label as i32;
        if let Some(pos) = label_list.iter().position(|&lab| lab == this_label) {
            let n = count.get_mut(pos)?;
            *n = n.checked_add(1)?;
            *slot = pos;
        } else {
            *slot = label_list.len();
            label_list.push(this_label);
            count.push(1);
        }
    }

    let nr_class = label_list.len();

    if label_list.as_slice() == [-1, 1] {
        label_list.swap(0, 1);
        count.swap(0, 1);
        for dl in &mut data_label {
            *dl ^= 1;
        }
    }

    let mut start = vec![0usize; nr_class];
    let mut cumulative = 0usize;
    for (slot, &n) in start.iter_mut().zip(&count) {
        *slot = cumulative;
        cumulative = cumulative.checked_add(n)?;
    }

    let mut perm = vec![0usize; l];
    let mut start_copy = start.clone();
    for (i, &cls) in data_label.iter().enumerate() {
        let offset = start_copy.get_mut(cls)?;
        *perm.get_mut(*offset)? = i;
        *offset = offset.checked_add(1)?;
    }

    Some(GroupedClasses {
        label: label_list,
        start,
        count,
        perm,
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn c_rand() -> usize {
    static STATE: OnceLock<Mutex<u32>> = OnceLock::new();
    let state = STATE.get_or_init(|| Mutex::new(1));
    let mut guard = state
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    // macOS/BSD libc rand() uses Park-Miller (MINSTD) with 31-bit modulus.
    let hi = *guard / 127_773;
    let lo = *guard % 127_773;
    // hi and lo come from a u32; both products and their difference fit i64.
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "u32 inputs times 16807 or 2836, then subtracted, fit i64"
    )]
    let test = 16_807_i64 * i64::from(lo) - 2_836_i64 * i64::from(hi);
    *guard = if test > 0 {
        test as u32
    } else {
        // test is nonpositive and no smaller than -2836 * (u32::MAX / 127773).
        #[expect(
            clippy::arithmetic_side_effects,
            reason = "test is between -96000000 and zero; adding the modulus fits i64"
        )]
        let shifted = test + 2_147_483_647;
        shifted as u32
    };
    *guard as usize
}

#[cfg(target_os = "linux")]
const GLIBC_RAND_DEGREE: usize = 31;
#[cfg(target_os = "linux")]
const GLIBC_RAND_SEPARATION: usize = 3;

#[cfg(target_os = "linux")]
struct GlibcRandState {
    state: [u32; GLIBC_RAND_DEGREE],
    front: usize,
    rear: usize,
}

#[cfg(target_os = "linux")]
impl GlibcRandState {
    fn seeded(seed: u32) -> Option<Self> {
        let mut state = [0u32; GLIBC_RAND_DEGREE];
        state[0] = seed;
        // A u32 times 16807 is below 2^47, so the u64 product cannot overflow.
        let mut previous = seed;
        for slot in state.iter_mut().skip(1) {
            *slot = ((16_807u64 * u64::from(previous)) % 2_147_483_647) as u32;
            previous = *slot;
        }

        let mut rng = Self {
            state,
            front: GLIBC_RAND_SEPARATION,
            rear: 0,
        };
        for _ in 0..(10 * GLIBC_RAND_DEGREE) {
            rng.next()?;
        }
        Some(rng)
    }

    fn next(&mut self) -> Option<usize> {
        let rear = *self.state.get(self.rear)?;
        let front = self.state.get_mut(self.front)?;
        // glibc adds unsigned words modulo 2^32.
        *front = front.overflowing_add(rear).0;
        let value = (*front >> 1) & 0x7fff_ffff;
        self.front = self.front.checked_add(1)? % GLIBC_RAND_DEGREE;
        self.rear = self.rear.checked_add(1)? % GLIBC_RAND_DEGREE;
        Some(value as usize)
    }
}

#[cfg(target_os = "linux")]
fn glibc_rand_state() -> Option<&'static Mutex<GlibcRandState>> {
    static STATE: OnceLock<Option<Mutex<GlibcRandState>>> = OnceLock::new();
    STATE
        .get_or_init(|| GlibcRandState::seeded(1).map(Mutex::new))
        .as_ref()
}

#[cfg(target_os = "linux")]
pub(crate) fn c_rand() -> usize {
    // seeded initializes both offsets below 31; next keeps them below 31.
    #[expect(
        clippy::expect_used,
        reason = "seeded and next keep the offsets below the 31-word state length"
    )]
    glibc_rand_state()
        .and_then(|state| {
            state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .next()
        })
        .expect("random state offsets must be below 31; this is a library bug, report it")
}

/// Deterministic fallback PRNG used where libc rand() parity is not implemented.
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub(crate) fn c_rand() -> usize {
    static STATE: OnceLock<Mutex<u64>> = OnceLock::new();
    let state = STATE.get_or_init(|| Mutex::new(1));
    let mut guard = state
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    // This generator uses arithmetic modulo 2^64.
    *guard = guard
        .overflowing_mul(6364136223846793005)
        .0
        .overflowing_add(1442695040888963407)
        .0;
    (*guard >> 33) as usize
}

/// Shuffle exactly `len` entries from `index[start..start + len]` in place.
///
/// Uses the platform-scoped libc `rand()` replica so public k-fold CV consumes
/// the same shuffle source as LIBSVM's `svm_cross_validation`.
pub(crate) fn shuffle_range(index: &mut [usize], start: usize, len: usize) -> Option<()> {
    let end = start.checked_add(len)?;
    let slice = index.get_mut(start..end)?;
    for i in 0..len {
        let remaining = len.checked_sub(i)?;
        let j = i.checked_add(c_rand().checked_rem(remaining)?)?;
        slice.swap(i, j);
    }
    Some(())
}

/// Parse and validate a feature index according to the shared hard limit.
#[expect(
    clippy::if_not_else,
    reason = "Keep the C branch order for comparison with the original"
)]
/// # Errors
///
/// Returns an error string when the index is not an integer in range.
pub fn parse_feature_index(idx_str: &str, max_feature_index: i32) -> Result<i32, String> {
    let index = idx_str
        .parse::<i32>()
        .map_err(|_| format!("invalid feature index: {idx_str}"))?;

    if !(0..=max_feature_index).contains(&index) {
        Err(format!(
            "feature index {index} exceeds limit ({max_feature_index})"
        ))
    } else {
        Ok(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "linux")]
    #[test]
    #[expect(
        clippy::indexing_slicing,
        reason = "The small fixed fixture bounds each index"
    )]
    fn cross_validation_single_member_class_draws_one_random_number() {
        // Run alone so other tests cannot consume the process-wide random stream.
        const CHILD: &str = "LIBSVM_SINGLE_MEMBER_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "util::tests::cross_validation_single_member_class_draws_one_random_number",
                    "--nocapture",
                ])
                .env(CHILD, "1")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }

        let labels = vec![7.0, 2.0, 2.0, 2.0, 2.0, -3.0, -3.0, -3.0, -3.0];
        let starts = [0, 1, 5];
        let counts = [1, 4, 4];
        let mut rng = GlibcRandState::seeded(1).unwrap();
        let mut index: Vec<_> = (0..labels.len()).collect();
        // Independent copy of the C shuffle, including rand() % 1.
        for (&start, &len) in starts.iter().zip(&counts) {
            for i in 0..len {
                let j = i + rng.next().unwrap() % (len - i);
                index.swap(start + i, start + j);
            }
        }
        let mut expected = Vec::new();
        let mut fold_starts = vec![0];
        for fold in 0..3 {
            for (&start, &len) in starts.iter().zip(&counts) {
                expected.extend_from_slice(
                    &index[start + fold * len / 3..start + (fold + 1) * len / 3],
                );
            }
            fold_starts.push(expected.len());
        }
        let param = crate::SvmParameter {
            kernel_type: crate::KernelType::Linear,
            ..Default::default()
        };
        let (actual, actual_starts) =
            crate::cross_validation::cross_validation_folds(&labels, &param, 3).unwrap();
        assert_eq!(actual, expected, "stratified permutation differs from C");
        assert_eq!(actual_starts, fold_starts);
        shuffle_range(&mut [], 0, 0).unwrap();
        assert_eq!(
            c_rand(),
            rng.next().unwrap(),
            "random stream differs after the shuffle"
        );

        // Exercise the public call from the same seed and independent fold plan.
        *glibc_rand_state().unwrap().lock().unwrap() = GlibcRandState::seeded(1).unwrap();
        let problem = crate::SvmProblem {
            instances: (0..9)
                .map(|i| {
                    vec![crate::SvmNode {
                        index: 1,
                        value: f64::from(i),
                    }]
                })
                .collect(),
            labels,
        };
        let mut expected_targets = [0.0; 9];
        for bounds in fold_starts.windows(2) {
            let (begin, end) = (bounds[0], bounds[1]);
            let training: Vec<_> = expected[..begin]
                .iter()
                .chain(&expected[end..])
                .copied()
                .collect();
            let subproblem = crate::SvmProblem {
                labels: training.iter().map(|&i| problem.labels[i]).collect(),
                instances: training
                    .iter()
                    .map(|&i| problem.instances[i].clone())
                    .collect(),
            };
            let model = crate::train::svm_train(&subproblem, &param);
            for &i in &expected[begin..end] {
                expected_targets[i] = crate::predict::predict(&model, &problem.instances[i]);
            }
        }
        // The comparison is exact (zero tolerance). Each target is a predicted class
        // label, an i32 converted to f64, and both sides run the same training.
        let actual_targets = crate::cross_validation::svm_cross_validation(&problem, &param, 3);
        assert_eq!(
            actual_targets
                .into_iter()
                .map(f64::to_bits)
                .collect::<Vec<_>>(),
            expected_targets.map(f64::to_bits)
        );
    }

    #[test]
    fn shuffle_range_rejects_inconsistent_window() {
        assert!(shuffle_range(&mut [0], 1, 1).is_none());
        assert!(shuffle_range(&mut [0], usize::MAX, 1).is_none());
        assert!(shuffle_range(&mut [0], 2, 0).is_none());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn random_next_rejects_invalid_offsets() {
        let mut rng = GlibcRandState::seeded(1).unwrap();
        rng.front = GLIBC_RAND_DEGREE;
        assert!(rng.next().is_none());
        rng.front = 0;
        rng.rear = GLIBC_RAND_DEGREE;
        assert!(rng.next().is_none());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn c_rand_matches_glibc_first_outputs() {
        let expected = [
            1_804_289_383usize,
            846_930_886,
            1_681_692_777,
            1_714_636_915,
            1_957_747_793,
            424_238_335,
            719_885_386,
            1_649_760_492,
            596_516_649,
            1_189_641_421,
            1_025_202_362,
            1_350_490_027,
            783_368_690,
            1_102_520_059,
            2_044_897_763,
            1_967_513_926,
            1_365_180_540,
            1_540_383_426,
            304_089_172,
            1_303_455_736,
        ];
        let mut rng = GlibcRandState::seeded(1).unwrap();
        let actual: Vec<_> = (0..expected.len()).map(|_| rng.next().unwrap()).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn group_classes_reorders_binary_negative_one_positive_one() {
        let labels = [-1.0, -1.0, 1.0, 1.0];
        let grouped = group_classes(&labels).unwrap();
        assert_eq!(grouped.label, vec![1, -1]);
        assert_eq!(grouped.count, vec![2, 2]);
        assert_eq!(grouped.start, vec![0, 2]);
        assert_eq!(grouped.perm.len(), labels.len());
    }

    #[test]
    fn group_classes_preserves_first_seen_order_for_other_labels() {
        let labels = [2.0, 1.0, 2.0, 3.0];
        let grouped = group_classes(&labels).unwrap();
        assert_eq!(grouped.label, vec![2, 1, 3]);
        assert_eq!(grouped.count, vec![2, 1, 1]);
    }

    #[test]
    fn parse_feature_index_rejects_non_integer() {
        let err = parse_feature_index("abc", 10).unwrap_err();
        assert_eq!(err, "invalid feature index: abc");
    }

    #[test]
    fn parse_feature_index_accepts_zero_and_maximum() {
        assert_eq!(parse_feature_index("0", 10).unwrap(), 0);
        assert_eq!(parse_feature_index("10", 10).unwrap(), 10);
    }

    #[test]
    fn parse_feature_index_rejects_negative_indices() {
        let err = parse_feature_index("-1", 10).unwrap_err();
        assert_eq!(err, "feature index -1 exceeds limit (10)");
    }

    #[test]
    fn parse_feature_index_rejects_out_of_range() {
        let err = parse_feature_index("11", 10).unwrap_err();
        assert_eq!(err, "feature index 11 exceeds limit (10)");
    }

    #[test]
    fn shuffle_range_keeps_length_and_determinism() {
        let mut order = vec![0, 1, 2, 3, 4];
        shuffle_range(&mut order, 1, 3).unwrap();
        assert_eq!(order.len(), 5);
        assert_eq!(*order.first().unwrap(), 0);
        assert_eq!(*order.get(4).unwrap(), 4);
        let mut window = order.get(1..4).unwrap().to_vec();
        window.sort_unstable();
        assert_eq!(window, vec![1, 2, 3]);
    }

    #[test]
    fn shuffle_range_does_not_change_values_outside_window() {
        let mut order = vec![10, 11, 12, 13, 14];
        let prefix = [10];
        let suffix = [14];

        shuffle_range(&mut order, 1, 3).unwrap();
        assert_eq!(order.get(..1).unwrap(), prefix);
        assert_eq!(order.get(4..).unwrap(), suffix);
        assert_eq!(order.len(), 5);
    }
}
