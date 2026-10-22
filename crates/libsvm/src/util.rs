//! Shared internal utilities for libsvm-rs.

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
pub(crate) fn group_classes(labels: &[f64]) -> GroupedClasses {
    let l = labels.len();
    let mut label_list: Vec<i32> = Vec::new();
    let mut count: Vec<usize> = Vec::new();
    let mut data_label = vec![0usize; l];

    for (i, &label) in labels.iter().enumerate() {
        let this_label = label as i32;
        if let Some(pos) = label_list.iter().position(|&lab| lab == this_label) {
            count[pos] += 1;
            data_label[i] = pos;
        } else {
            data_label[i] = label_list.len();
            label_list.push(this_label);
            count.push(1);
        }
    }

    let nr_class = label_list.len();

    if nr_class == 2 && label_list[0] == -1 && label_list[1] == 1 {
        label_list.swap(0, 1);
        count.swap(0, 1);
        for dl in data_label.iter_mut() {
            *dl ^= 1;
        }
    }

    let mut start = vec![0usize; nr_class];
    for i in 1..nr_class {
        start[i] = start[i - 1] + count[i - 1];
    }

    let mut perm = vec![0usize; l];
    let mut start_copy = start.clone();
    for (i, &cls) in data_label.iter().enumerate() {
        perm[start_copy[cls]] = i;
        start_copy[cls] += 1;
    }

    GroupedClasses {
        label: label_list,
        start,
        count,
        perm,
    }
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
    let test = 16_807_i64 * lo as i64 - 2_836_i64 * hi as i64;
    *guard = if test > 0 {
        test as u32
    } else {
        (test + 2_147_483_647) as u32
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
    fn seeded(seed: u32) -> Self {
        let mut state = [0u32; GLIBC_RAND_DEGREE];
        state[0] = seed;
        for i in 1..GLIBC_RAND_DEGREE {
            state[i] = ((16_807u64 * state[i - 1] as u64) % 2_147_483_647) as u32;
        }

        let mut rng = Self {
            state,
            front: GLIBC_RAND_SEPARATION,
            rear: 0,
        };
        for _ in 0..(10 * GLIBC_RAND_DEGREE) {
            rng.next();
        }
        rng
    }

    fn next(&mut self) -> usize {
        self.state[self.front] = self.state[self.front].wrapping_add(self.state[self.rear]);
        let value = (self.state[self.front] >> 1) & 0x7fff_ffff;
        self.front = (self.front + 1) % GLIBC_RAND_DEGREE;
        self.rear = (self.rear + 1) % GLIBC_RAND_DEGREE;
        value as usize
    }
}

#[cfg(target_os = "linux")]
fn glibc_rand_state() -> &'static Mutex<GlibcRandState> {
    static STATE: OnceLock<Mutex<GlibcRandState>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(GlibcRandState::seeded(1)))
}

#[cfg(target_os = "linux")]
pub(crate) fn c_rand() -> usize {
    glibc_rand_state()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .next()
}

/// Deterministic fallback PRNG used where libc rand() parity is not implemented.
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub(crate) fn c_rand() -> usize {
    static STATE: OnceLock<Mutex<u64>> = OnceLock::new();
    let state = STATE.get_or_init(|| Mutex::new(1));
    let mut guard = state
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    *guard = guard
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*guard >> 33) as usize
}

/// Shuffle exactly `len` entries from `index[start..start + len]` in place.
///
/// Uses the platform-scoped libc `rand()` replica so public k-fold CV consumes
/// the same shuffle source as LIBSVM's `svm_cross_validation`.
pub(crate) fn shuffle_range(index: &mut [usize], start: usize, len: usize) {
    if len <= 1 {
        return;
    }
    let end = start + len;
    let slice = &mut index[start..end];
    for i in 0..len {
        let j = i + c_rand() % (len - i);
        slice.swap(i, j);
    }
}

/// Parse and validate a feature index according to the shared hard limit.
pub fn parse_feature_index(idx_str: &str, max_feature_index: i32) -> Result<i32, String> {
    let index = idx_str
        .parse::<i32>()
        .map_err(|_| format!("invalid feature index: {}", idx_str))?;

    if !(0..=max_feature_index).contains(&index) {
        Err(format!(
            "feature index {} exceeds limit ({})",
            index, max_feature_index
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
        let mut rng = GlibcRandState::seeded(1);
        let actual: Vec<_> = (0..expected.len()).map(|_| rng.next()).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn group_classes_reorders_binary_negative_one_positive_one() {
        let labels = [-1.0, -1.0, 1.0, 1.0];
        let grouped = group_classes(&labels);
        assert_eq!(grouped.label, vec![1, -1]);
        assert_eq!(grouped.count, vec![2, 2]);
        assert_eq!(grouped.start, vec![0, 2]);
        assert_eq!(grouped.perm.len(), labels.len());
    }

    #[test]
    fn group_classes_preserves_first_seen_order_for_other_labels() {
        let labels = [2.0, 1.0, 2.0, 3.0];
        let grouped = group_classes(&labels);
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
        shuffle_range(&mut order, 1, 3);
        assert_eq!(order.len(), 5);
        assert_eq!(order[0], 0);
        assert_eq!(order[4], 4);
        let mut window = order[1..4].to_vec();
        window.sort_unstable();
        assert_eq!(window, vec![1, 2, 3]);
    }

    #[test]
    fn shuffle_range_does_not_change_values_outside_window() {
        let mut order = vec![10, 11, 12, 13, 14];
        let prefix = [10];
        let suffix = [14];

        shuffle_range(&mut order, 1, 3);
        assert_eq!(&order[..1], prefix);
        assert_eq!(&order[4..], suffix);
        assert_eq!(order.len(), 5);
    }
}
