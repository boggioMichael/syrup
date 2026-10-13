//! Scanning byte rows for runs of non-zero values: the inner loop shared by
//! connected-component labelling and motion extraction.

/// Call `found(start, end)` for every maximal run of non-zero bytes in
/// `row`, as a half-open range, in increasing order. Background is skipped
/// eight bytes at a time, so sparse rows cost little more than reading them.
pub(crate) fn for_each_run(row: &[u8], mut found: impl FnMut(usize, usize)) {
    let mut x = 0;
    while x < row.len() {
        match first_nonzero(&row[x..]) {
            Some(offset) => x += offset,
            None => break,
        }
        let end = row[x..]
            .iter()
            .position(|&v| v == 0)
            .map_or(row.len(), |offset| x + offset);
        found(x, end);
        x = end;
    }
}

/// Index of the first non-zero byte, testing eight at a time.
pub(crate) fn first_nonzero(bytes: &[u8]) -> Option<usize> {
    let (chunks, remainder) = bytes.as_chunks::<8>();
    let mut offset = 0;
    for chunk in chunks {
        if u64::from_ne_bytes(*chunk) != 0 {
            return chunk.iter().position(|&v| v != 0).map(|i| offset + i);
        }
        offset += 8;
    }
    remainder.iter().position(|&v| v != 0).map(|i| offset + i)
}

/// Call `found(start, end)` for every maximal run of set bits among the
/// first `len` bits of `bits` (bit `i % 64` of word `i / 64`; the bits past
/// `len` must be clear), as a half-open range, in increasing order: each
/// run's ends found a word at a time.
pub(crate) fn for_each_bit_run(bits: &[u64], len: usize, mut found: impl FnMut(usize, usize)) {
    debug_assert!(bits.len() * 64 >= len);
    let mut x = 0;
    while x < len {
        let start = next_bit(bits, x, len, 0);
        if start >= len {
            break;
        }
        let end = next_bit(bits, start, len, u64::MAX);
        found(start, end);
        x = end;
    }
}

/// The first bit at or after `from` that is set in `bits ^ flip` (`flip` 0:
/// a set bit; all ones: a clear one), or `len` when none is before it.
fn next_bit(bits: &[u64], from: usize, len: usize, flip: u64) -> usize {
    let mut w = from / 64;
    let mut word = (bits[w] ^ flip) & (u64::MAX << (from % 64));
    loop {
        if word != 0 {
            return (w * 64 + word.trailing_zeros() as usize).min(len);
        }
        w += 1;
        if w * 64 >= len {
            return len;
        }
        word = bits[w] ^ flip;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runs(row: &[u8]) -> Vec<(usize, usize)> {
        let mut found = Vec::new();
        for_each_run(row, |a, b| found.push((a, b)));
        found
    }

    #[test]
    fn runs_of_bits_are_the_runs_of_bytes() {
        let mut state = 7u32;
        for len in (0..200usize).chain([640, 1000]) {
            for density in [1u32, 3, 6, 9] {
                let row: Vec<u8> = (0..len)
                    .map(|_| {
                        state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
                        u8::from((state >> 16) % 10 < density)
                    })
                    .collect();
                let mut bits = vec![0u64; len.div_ceil(64)];
                for (i, &v) in row.iter().enumerate() {
                    bits[i / 64] |= u64::from(v) << (i % 64);
                }
                let mut found = Vec::new();
                for_each_bit_run(&bits, len, |a, b| found.push((a, b)));
                assert_eq!(found, runs(&row), "{len} bits at {density}0%");
            }
        }
    }

    #[test]
    fn finds_every_run_including_at_the_edges() {
        let mut row = vec![0u8; 37];
        row[0] = 1;
        row[8] = 3;
        row[20..30].fill(1);
        row[36] = 9;
        assert_eq!(runs(&row), vec![(0, 1), (8, 9), (20, 30), (36, 37)]);
        assert_eq!(runs(&[5; 11]), vec![(0, 11)]);
        assert!(runs(&[0; 17]).is_empty());
        assert!(runs(&[]).is_empty());
    }

    #[test]
    fn matches_a_plain_scan() {
        let mut state = 7u32;
        for len in 0..70 {
            let row: Vec<u8> = (0..len)
                .map(|_| {
                    state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
                    u8::from((state >> 16).is_multiple_of(3))
                })
                .collect();
            let mut expected = Vec::new();
            let mut start = None;
            for (x, &v) in row.iter().enumerate() {
                match (v != 0, start) {
                    (true, None) => start = Some(x),
                    (false, Some(s)) => {
                        expected.push((s, x));
                        start = None;
                    }
                    _ => {}
                }
            }
            if let Some(s) = start {
                expected.push((s, row.len()));
            }
            assert_eq!(runs(&row), expected);
            assert_eq!(first_nonzero(&row), row.iter().position(|&v| v != 0));
        }
    }
}
