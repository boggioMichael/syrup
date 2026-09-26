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
    let mut chunks = bytes.chunks_exact(8);
    let mut offset = 0;
    for chunk in &mut chunks {
        if u64::from_ne_bytes(chunk.try_into().expect("8 bytes")) != 0 {
            return chunk.iter().position(|&v| v != 0).map(|i| offset + i);
        }
        offset += 8;
    }
    chunks
        .remainder()
        .iter()
        .position(|&v| v != 0)
        .map(|i| offset + i)
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
