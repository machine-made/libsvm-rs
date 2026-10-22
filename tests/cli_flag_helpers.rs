pub fn xorshift64star(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

pub fn shuffle_flag_chunks<T>(chunks: &mut [Vec<T>], state: &mut u64) {
    let len = chunks.len();
    if len <= 1 {
        return;
    }
    for i in (1..len).rev() {
        let j = (xorshift64star(state) as usize) % (i + 1);
        chunks.swap(i, j);
    }
}
