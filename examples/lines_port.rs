//! `syrup::lines` on the extraction pilot's lines (`bench/extract/a/export.py`
//! writes them; they are game graphics, kept outside the repository).
//!
//! ```text
//! cargo run --release --example lines_port -- DIR [time]
//! ```
//!
//! Learns a [`LineFont`] from the pilot's training lines (made from the
//! training glyphs, and real frames of the training runs), reads the test
//! lines, and says how many it read right and wrong and refused: refusing
//! by the margin alone ([`LineFont::learn`]), and with the EXP form checked
//! first ([`LineFont::learn_checked`], the pilot's rule). With `time`, each test line is read 3 times, then 5 times
//! timed; a line's time is the median, and p50 and p95 are over the lines.

use std::fs;
use std::path::Path;
use std::time::Instant;

use image::RgbaImage;
use syrup::geometry::Rect;
use syrup::lines::{LabelledLine, LineFont, LineOptions};

const ROWS: usize = 15;

/// An EXP value as the HUD writes it: digits in threes with commas, a
/// bracket, a percentage with two or three decimals, the closing bracket.
fn is_exp(text: &str) -> bool {
    let Some((value, rest)) = text.split_once('[') else {
        return false;
    };
    let groups: Vec<&str> = value.split(',').collect();
    let value_ok = !groups[0].is_empty()
        && groups[0].len() <= 3
        && groups.iter().all(|g| g.bytes().all(|b| b.is_ascii_digit()))
        && groups[1..].iter().all(|g| g.len() == 3);
    let Some(pct) = rest.strip_suffix("%]") else {
        return false;
    };
    let Some((whole, frac)) = pct.split_once('.') else {
        return false;
    };
    value_ok
        && (1..=3).contains(&whole.len())
        && (2..=3).contains(&frac.len())
        && whole
            .bytes()
            .chain(frac.bytes())
            .all(|b| b.is_ascii_digit())
}

fn lines(dir: &Path, prefix: &str, count: usize, w: usize) -> Vec<RgbaImage> {
    let px = fs::read(dir.join(format!("{prefix}lines.u8"))).expect("lines");
    let n = ROWS * w * 3;
    assert_eq!(px.len(), count * n);
    px.chunks_exact(n)
        .map(|line| {
            let mut img = RgbaImage::new(w as u32, ROWS as u32);
            for (to, p) in img
                .as_mut()
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(line.as_chunks::<3>().0)
            {
                to.copy_from_slice(&[p[0], p[1], p[2], 255]);
            }
            img
        })
        .collect()
}

fn text_lines(dir: &Path, file: &str) -> Vec<String> {
    fs::read_to_string(dir.join(file))
        .expect(file)
        .lines()
        .map(String::from)
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(args.get(1).expect("usage: lines_port DIR [time]"));
    let meta = fs::read_to_string(dir.join("meta.txt")).expect("meta.txt");
    let field = |key: &str| -> Vec<usize> {
        meta.lines()
            .find(|l| l.starts_with(&format!("{key} ")))
            .expect(key)
            .split_whitespace()
            .skip(1)
            .map(|v| v.parse().unwrap())
            .collect()
    };
    let (count, w) = (field("lines")[0], field("lines")[2]);
    let train_count = field("train")[0];
    let region = Rect {
        x: 0,
        y: 0,
        w: w as u32,
        h: ROWS as u32,
    };

    let train = lines(dir, "train-", train_count, w);
    let train_labels = text_lines(dir, "train-truth.txt");
    let labelled: Vec<LabelledLine> = train
        .iter()
        .zip(&train_labels)
        .map(|(image, label)| LabelledLine {
            image,
            region,
            label,
        })
        .collect();
    let t0 = Instant::now();
    let plain = LineFont::learn(&labelled, LineOptions::default()).expect("learns");
    println!(
        "learned from {} lines in {:.1} s: {} characters, refusal level {:.3} (the margin alone)",
        labelled.len(),
        t0.elapsed().as_secs_f64(),
        plain.chars().len(),
        plain.level()
    );
    let font = LineFont::learn_checked(&labelled, LineOptions::default(), is_exp).expect("learns");
    println!(
        "with the EXP form checked: refusal level {:.3}",
        font.level()
    );

    let test = lines(dir, "", count, w);
    let truth = text_lines(dir, "truth.txt");
    let kind = text_lines(dir, "kind.txt");
    // [made, real] x [exact before refusing, right, wrong, refused]: the
    // margin's level alone, then the form checked too
    let mut tallies = [[[0usize; 4]; 2]; 2];
    for (f, which) in [&plain, &font].into_iter().zip(0..) {
        let t = &mut tallies[which];
        for (i, image) in test.iter().enumerate() {
            let k = usize::from(kind[i] == "real");
            let exact = f
                .read_all(image, region)
                .is_some_and(|r| r.text == truth[i]);
            t[k][0] += usize::from(exact);
            match f.read(image, region).value {
                Some(r) => t[k][if r.text == truth[i] { 1 } else { 2 }] += 1,
                None => t[k][3] += 1,
            }
        }
    }
    for (name, t) in [
        ("the margin alone", tallies[0]),
        ("the form, then the margin", tallies[1]),
    ] {
        println!(
            "{name}: made exact {} | right/wrong/refused {}/{}/{}; real exact {} | {}/{}/{}",
            t[0][0], t[0][1], t[0][2], t[0][3], t[1][0], t[1][1], t[1][2], t[1][3]
        );
    }

    if args.get(2).map(String::as_str) == Some("time") {
        let mut per = [Vec::new(), Vec::new()];
        let mut sink = 0usize;
        for (i, image) in test.iter().enumerate() {
            for _ in 0..3 {
                sink += font.read_all(image, region).map_or(0, |r| r.text.len());
            }
            let mut ns = Vec::new();
            for _ in 0..5 {
                let t0 = Instant::now();
                sink += font.read_all(image, region).map_or(0, |r| r.text.len());
                ns.push(t0.elapsed().as_nanos() as u64);
            }
            ns.sort_unstable();
            per[usize::from(kind[i] == "real")].push(ns[2]);
        }
        for (name, v) in ["made", "real"].iter().zip(per.iter_mut()) {
            v.sort_unstable();
            let p = |q: f64| v[((v.len() - 1) as f64 * q).round() as usize] as f64 / 1e3;
            println!("{name}: p50 {:.1} us, p95 {:.1} us", p(0.5), p(0.95));
        }
        eprintln!("({sink})");
    }
}
