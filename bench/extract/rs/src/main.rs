//! Pilot A's readers in Rust, for what each costs a line (PILOT-A.md).
//!
//! ```text
//! extract-port check DIR            the port reads what the Python code read
//! extract-port time DIR READER [WARMUP CALLS [pre]]
//!                                   per-line times: READER is teacher,
//!                                   white-box, black-box or syrup (Syrup's
//!                                   glyph reader, `glyphs.rs`); `pre`: the
//!                                   line normalised before the clock starts
//! ```
//!
//! DIR is what `a/export.py` wrote. Both kinds of reader start from the same
//! RGB line (15 x 210) and do all of their own work: the line normalised as
//! the teacher was trained on (`linenet.to_input`), then
//!
//! - the teacher: five convolutions (93,857 numbers, 37.4M multiply-adds a
//!   line), ReLU, a 2x2 max-pool, and CTC's greedy decoding;
//! - an explicit reader: the line made grey, every class's picture compared
//!   at every column, the cheapest cover found by dynamic programming, each
//!   placed glyph read again by 17 linear templates over its 15 x 9 window
//!   (3,227 numbers, ~0.25M operations a line).
//!
//! Plain loops, no SIMD intrinsics, no threads: what the compiler makes of
//! them (`-C target-cpu=native` when timed).

use std::fs;
use std::path::Path;
use std::time::Instant;

mod glyphs;

const ALPHABET: &[u8] = b"0123456789,./[]%";
const ROWS: usize = 15; // the line's rows
const H: usize = 16; // the teacher's input: the 15 rows and the last one again
const W: usize = 216; // columns the readers see (the line's 210 and 6 of padding)
const NCLASS: usize = 17;

// ------------------------------------------------------------------ data

struct Meta {
    lines: usize,
    train: usize,
    line_w: usize,
    layers: Vec<[usize; 6]>,
    readers: Vec<(String, f64, usize, Vec<(usize, usize)>)>, // name, penalty, hw, (class, width)
}

fn parse_meta(text: &str) -> Meta {
    let mut m = Meta { lines: 0, train: 0, line_w: 0, layers: vec![], readers: vec![] };
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        match f.first().copied() {
            Some("lines") => {
                m.lines = f[1].parse().unwrap();
                assert_eq!(f[2], "15");
                m.line_w = f[3].parse().unwrap();
            }
            Some("train") => m.train = f[1].parse().unwrap(),
            Some("layer") => {
                let v: Vec<usize> = f[1..7].iter().map(|s| s.parse().unwrap()).collect();
                m.layers.push([v[0], v[1], v[2], v[3], v[4], v[5]]);
            }
            Some("reader") => m.readers.push((f[1].to_string(), f[3].parse().unwrap(), f[4].parse().unwrap(), vec![])),
            Some("class") => {
                let r = m.readers.iter_mut().find(|r| r.0 == f[1]).unwrap();
                r.3.push((f[2].parse().unwrap(), f[3].parse().unwrap()));
            }
            _ => {}
        }
    }
    m
}

fn f32s(path: &Path) -> Vec<f32> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

// ------------------------------------------------------- the line, normalised

/// `linenet.to_input` for one line: RGB / 255, less each colour's mean, over
/// the spread of all of it plus 0.02; 16 rows (the last one twice) by 216
/// columns (zeros past the line), NHWC.
fn normalise(px: &[u8], line_w: usize, out: &mut [f32]) {
    let n = ROWS * line_w;
    let mut mean = [0f64; 3];
    for p in px.chunks_exact(3) {
        for c in 0..3 {
            mean[c] += p[c] as f64;
        }
    }
    for m in &mut mean {
        *m /= n as f64 * 255.0;
    }
    let (mut s, mut s2) = (0f64, 0f64);
    for p in px.chunks_exact(3) {
        for c in 0..3 {
            let v = p[c] as f64 / 255.0 - mean[c];
            s += v;
            s2 += v * v;
        }
    }
    let count = (n * 3) as f64;
    let m = s / count;
    let scale = 1.0 / (((s2 / count - m * m).max(0.0)).sqrt() + 0.02);
    out.fill(0.0);
    for r in 0..ROWS {
        for x in 0..line_w.min(W) {
            for c in 0..3 {
                let v = ((px[(r * line_w + x) * 3 + c] as f64 / 255.0 - mean[c]) * scale) as f32;
                out[(r * W + x) * 3 + c] = v;
                if r == ROWS - 1 {
                    out[((r + 1) * W + x) * 3 + c] = v;
                }
            }
        }
    }
}

// ------------------------------------------------------------- the teacher

struct Conv {
    cin: usize,
    cout: usize,
    kh: usize,
    kw: usize,
    ph: usize,
    pw: usize,
    w: Vec<f32>, // (kh*kw*cin, cout)
    b: Vec<f32>,
}

impl Conv {
    /// Zero-padded convolution, NHWC: each output position's scores start
    /// at the bias and take each input value times its row of weights (the
    /// innermost loop runs over the output channels, contiguous on both
    /// sides, which the compiler vectorises).
    fn forward(&self, x: &[f32], h: usize, w: usize, out: &mut Vec<f32>) -> (usize, usize) {
        let (cin, cout, kh, kw, ph, pw) = (self.cin, self.cout, self.kh, self.kw, self.ph, self.pw);
        let ho = h + 2 * ph - kh + 1;
        let wo = w + 2 * pw - kw + 1;
        out.clear();
        out.resize(ho * wo * cout, 0.0);
        for oy in 0..ho {
            for ox in 0..wo {
                let acc = &mut out[(oy * wo + ox) * cout..][..cout];
                acc.copy_from_slice(&self.b);
                for ky in 0..kh {
                    let iy = oy + ky;
                    if iy < ph || iy >= h + ph {
                        continue;
                    }
                    let iy = iy - ph;
                    for kx in 0..kw {
                        let ix = ox + kx;
                        if ix < pw || ix >= w + pw {
                            continue;
                        }
                        let ix = ix - pw;
                        let xin = &x[(iy * w + ix) * cin..][..cin];
                        let base = (ky * kw + kx) * cin;
                        for (ci, &v) in xin.iter().enumerate() {
                            let row = &self.w[(base + ci) * cout..][..cout];
                            for (a, &wv) in acc.iter_mut().zip(row) {
                                *a += v * wv;
                            }
                        }
                    }
                }
            }
        }
        (ho, wo)
    }
}

fn relu(x: &mut [f32]) {
    for v in x {
        *v = v.max(0.0);
    }
}

fn pool2(x: &[f32], h: usize, w: usize, c: usize, out: &mut Vec<f32>) {
    out.clear();
    out.resize((h / 2) * (w / 2) * c, 0.0);
    for y in 0..h / 2 {
        for xx in 0..w / 2 {
            for ch in 0..c {
                let at = |yy: usize, xi: usize| x[(yy * w + xi) * c + ch];
                let m = at(2 * y, 2 * xx).max(at(2 * y, 2 * xx + 1)).max(at(2 * y + 1, 2 * xx)).max(at(2 * y + 1, 2 * xx + 1));
                out[(y * (w / 2) + xx) * c + ch] = m;
            }
        }
    }
}

struct Teacher {
    convs: Vec<Conv>,
    a: Vec<f32>,
    b: Vec<f32>,
}

impl Teacher {
    fn load(dir: &Path, meta: &Meta) -> Teacher {
        let all = f32s(&dir.join("teacher.f32"));
        let mut at = 0;
        let mut convs = vec![];
        for &[cin, cout, kh, kw, ph, pw] in &meta.layers {
            let n = kh * kw * cin * cout;
            let w = all[at..at + n].to_vec();
            at += n;
            let b = all[at..at + cout].to_vec();
            at += cout;
            convs.push(Conv { cin, cout, kh, kw, ph, pw, w, b });
        }
        assert_eq!(at, all.len());
        Teacher { convs, a: vec![], b: vec![] }
    }

    /// conv, ReLU, conv, ReLU, pool, conv, ReLU, conv (all rows), ReLU, conv 1x1;
    /// then CTC's greedy decoding of the 108 columns.
    fn read(&mut self, x: &[f32]) -> String {
        let (mut h, mut w) = (H, W);
        let c = &self.convs;
        (h, w) = c[0].forward(x, h, w, &mut self.a);
        relu(&mut self.a);
        (h, w) = c[1].forward(&self.a, h, w, &mut self.b);
        relu(&mut self.b);
        pool2(&self.b, h, w, c[1].cout, &mut self.a);
        (h, w) = (h / 2, w / 2);
        (h, w) = c[2].forward(&self.a, h, w, &mut self.b);
        relu(&mut self.b);
        (h, w) = c[3].forward(&self.b, h, w, &mut self.a);
        relu(&mut self.a);
        (h, w) = c[4].forward(&self.a, h, w, &mut self.b);
        debug_assert_eq!(h, 1);
        let mut text = String::new();
        let mut prev = 0;
        for t in 0..w {
            let s = &self.b[t * NCLASS..][..NCLASS];
            let mut k = 0;
            for j in 1..NCLASS {
                if s[j] > s[k] {
                    k = j;
                }
            }
            if k != prev && k != 0 {
                text.push(ALPHABET[k - 1] as char);
            }
            prev = k;
        }
        text
    }
}

// ------------------------------------------------------- an explicit reader

struct Reader {
    classes: Vec<usize>,
    widths: Vec<usize>,
    pictures: Vec<Vec<f32>>, // ROWS x width, row-major
    band: [f32; ROWS],
    weights: Vec<f32>, // (ROWS * (2*hw+1) + 1) x NCLASS
    penalty: f64,
    hw: usize,
    // scratch
    g: Vec<f32>,
    cost: Vec<Vec<f32>>,
    best: Vec<f64>,
    back: Vec<(usize, usize)>, // (class index + 1, or 0 for the band; start column)
}

impl Reader {
    fn load(dir: &Path, meta: &Meta, name: &str) -> Reader {
        let (_, penalty, hw, cls) = meta.readers.iter().find(|r| r.0 == name).unwrap_or_else(|| panic!("no reader {name}"));
        let all = f32s(&dir.join(format!("{name}.f32")));
        let mut band = [0f32; ROWS];
        band.copy_from_slice(&all[..ROWS]);
        let mut at = ROWS;
        let mut pictures = vec![];
        for &(_, w) in cls {
            pictures.push(all[at..at + ROWS * w].to_vec());
            at += ROWS * w;
        }
        let weights = all[at..].to_vec();
        assert_eq!(weights.len(), (ROWS * (2 * hw + 1) + 1) * NCLASS);
        Reader {
            classes: cls.iter().map(|c| c.0).collect(),
            widths: cls.iter().map(|c| c.1).collect(),
            pictures,
            band,
            weights,
            penalty: *penalty,
            hw: *hw,
            g: vec![0.0; ROWS * W],
            cost: cls.iter().map(|_| vec![0.0; W]).collect(),
            best: vec![0.0; W + 1],
            back: vec![(0, 0); W + 1],
        }
    }

    /// (text, confidence) for a normalised line, as `extract.Reader.read`.
    fn read(&mut self, x: &[f32]) -> (String, f64) {
        // grey: the mean of the three colours, rows 0..15
        for r in 0..ROWS {
            for c in 0..W {
                let p = &x[(r * W + c) * 3..][..3];
                self.g[r * W + c] = (p[0] + p[1] + p[2]) / 3.0;
            }
        }
        let g = &self.g;
        // step 1, the placement: each class's picture against every stretch
        // of the line, the band against every column, the cheapest cover
        let mut bg = [0f32; W];
        for r in 0..ROWS {
            for c in 0..W {
                let d = g[r * W + c] - self.band[r];
                bg[c] += d * d;
            }
        }
        for (k, pic) in self.pictures.iter().enumerate() {
            let w = self.widths[k];
            let cost = &mut self.cost[k];
            cost.fill(0.0);
            let n = W - w + 1;
            for r in 0..ROWS {
                for j in 0..w {
                    let t = pic[r * w + j];
                    let row = &g[r * W + j..][..n];
                    for (c, &v) in cost[..n].iter_mut().zip(row) {
                        let d = v - t;
                        *c += d * d;
                    }
                }
            }
            cost[n..].fill(f32::INFINITY);
        }
        self.best[0] = 0.0;
        for x in 1..=W {
            let mut v = self.best[x - 1] + bg[x - 1] as f64;
            let mut arg = (0, x - 1);
            for k in 0..self.classes.len() {
                let w = self.widths[k];
                if x >= w {
                    let s = x - w;
                    let c = self.best[s] + self.cost[k][s] as f64 + self.penalty;
                    if c < v {
                        v = c;
                        arg = (k + 1, s);
                    }
                }
            }
            self.best[x] = v;
            self.back[x] = arg;
        }
        let mut placed = vec![];
        let mut x = W;
        while x > 0 {
            let (k, s) = self.back[x];
            if k > 0 {
                placed.push((k - 1, s));
            }
            x = s;
        }
        placed.reverse();
        if placed.is_empty() {
            return (String::new(), f64::NEG_INFINITY);
        }
        // step 2, the reading: each glyph's 15 x 9 window, 17 templates
        let span = 2 * self.hw + 1;
        let nf = ROWS * span + 1;
        let mut text = String::new();
        let mut conf = f64::INFINITY;
        let mut scores = [0f64; NCLASS];
        for &(k, s) in &placed {
            let centre = (s + self.widths[k] / 2) as isize;
            scores.copy_from_slice(&[0.0; NCLASS]);
            for r in 0..ROWS {
                for j in 0..span {
                    let col = centre - self.hw as isize + j as isize;
                    if col < 0 || col >= W as isize {
                        continue;
                    }
                    let v = g[r * W + col as usize] as f64;
                    let row = &self.weights[(r * span + j) * NCLASS..][..NCLASS];
                    for (a, &wv) in scores.iter_mut().zip(row) {
                        *a += v * wv as f64;
                    }
                }
            }
            let bias = &self.weights[(nf - 1) * NCLASS..][..NCLASS];
            for (a, &b) in scores.iter_mut().zip(bias) {
                *a += b as f64;
            }
            // a digit stays a digit, a mark a mark
            let class = self.classes[k];
            let group: &[usize] = if (1..=10).contains(&class) { &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10] } else { &[11, 12, 13, 14, 15, 16] };
            let (mut first, mut second) = (usize::MAX, usize::MAX);
            for &c in group {
                if first == usize::MAX || scores[c] > scores[first] {
                    second = first;
                    first = c;
                } else if second == usize::MAX || scores[c] > scores[second] {
                    second = c;
                }
            }
            text.push(ALPHABET[first - 1] as char);
            conf = conf.min(scores[first] - scores[second]);
        }
        (text, conf)
    }
}

// ----------------------------------------------------------------- running

struct Data {
    meta: Meta,
    px: Vec<u8>,
    truth: Vec<String>,
    kind: Vec<String>,
}

fn load(dir: &Path) -> Data {
    let meta = parse_meta(&fs::read_to_string(dir.join("meta.txt")).unwrap());
    let px = fs::read(dir.join("lines.u8")).unwrap();
    assert_eq!(px.len(), meta.lines * ROWS * meta.line_w * 3);
    let read_lines = |f: &str| fs::read_to_string(dir.join(f)).unwrap().lines().map(String::from).collect::<Vec<_>>();
    Data { truth: read_lines("truth.txt"), kind: read_lines("kind.txt"), meta, px }
}

impl Data {
    fn line(&self, i: usize) -> &[u8] {
        let n = ROWS * self.meta.line_w * 3;
        &self.px[i * n..][..n]
    }

    /// Syrup's glyph reader, the font learned from the training lines.
    fn glyphs(&self, dir: &Path, min_score: f32) -> glyphs::Glyphs {
        let px = fs::read(dir.join("train-lines.u8")).unwrap();
        let n = ROWS * self.meta.line_w * 3;
        assert_eq!(px.len(), self.meta.train * n);
        let truth = fs::read_to_string(dir.join("train-truth.txt")).unwrap();
        let labels: Vec<&str> = truth.lines().collect();
        glyphs::Glyphs::learn(px.chunks_exact(n).zip(labels.iter().copied()), self.meta.line_w, ROWS, min_score)
    }
}

fn check(dir: &Path) {
    let d = load(dir);
    let mut x = vec![0f32; H * W * 3];
    let mut teacher = Teacher::load(dir, &d.meta);
    let py: Vec<String> = fs::read_to_string(dir.join("py-teacher.txt")).unwrap().lines().map(String::from).collect();
    let (mut same, mut right) = (0, [0usize; 2]);
    for i in 0..d.meta.lines {
        normalise(d.line(i), d.meta.line_w, &mut x);
        let t = teacher.read(&x);
        same += (t == py[i]) as usize;
        right[(d.kind[i] == "real") as usize] += (t == d.truth[i]) as usize;
    }
    println!("teacher: {same}/{} lines read as the Python code read them; exact: made {}, real {}", d.meta.lines, right[0], right[1]);
    for (name, ..) in &d.meta.readers {
        let mut r = Reader::load(dir, &d.meta, name);
        let py: Vec<(String, f64)> = fs::read_to_string(dir.join(format!("py-{name}.tsv")))
            .unwrap()
            .lines()
            .map(|l| {
                let (t, c) = l.split_once('\t').unwrap();
                (t.to_string(), c.parse::<f64>().unwrap_or(f64::NEG_INFINITY))
            })
            .collect();
        let (mut same, mut worst, mut right) = (0, 0f64, [0usize; 2]);
        for i in 0..d.meta.lines {
            normalise(d.line(i), d.meta.line_w, &mut x);
            let (t, c) = r.read(&x);
            if t == py[i].0 {
                same += 1;
                if c.is_finite() && py[i].1.is_finite() {
                    worst = worst.max((c - py[i].1).abs());
                }
            }
            right[(d.kind[i] == "real") as usize] += (t == d.truth[i]) as usize;
        }
        println!(
            "{name}: {same}/{} lines read as the Python code read them (confidence within {worst:.2e}); exact: made {}, real {}",
            d.meta.lines, right[0], right[1]
        );
    }
    // Syrup today: right, wrong and refused, made and real; with
    // MapleSyrup's options, and with the general default's looser score
    for (label, min_score) in [("syrup glyphs (MapleSyrup's options)", glyphs::MIN_SCORE), ("syrup glyphs (default options)", 0.72)] {
    let g = d.glyphs(dir, min_score);
    println!(
        "{label}: learned from {} of {} training lines (refused: no text {}, glyph count {}, inconsistent {}, does not read back {}); {} characters",
        g.learned,
        d.meta.train,
        g.refused[0],
        g.refused[1],
        g.refused[2],
        g.refused[3],
        g.set.chars().count()
    );
    let region = glyphs::whole(d.meta.line_w, ROWS);
    let mut tally = [[0usize; 3]; 2];
    for i in 0..d.meta.lines {
        let img = glyphs::rgba(d.line(i), d.meta.line_w, ROWS);
        let k = (d.kind[i] == "real") as usize;
        match g.read(&img, region) {
            Some(t) if t == d.truth[i] => tally[k][0] += 1,
            Some(_) => tally[k][1] += 1,
            None => tally[k][2] += 1,
        }
    }
    println!(
        "{label}: made right/wrong/refused {}/{}/{}, real {}/{}/{}",
        tally[0][0], tally[0][1], tally[0][2], tally[1][0], tally[1][1], tally[1][2]
    );
    // what it makes of a few lines, whatever its confidence
    if std::env::var_os("SHOW").is_some() {
        for i in [0usize, 1, d.meta.lines - 1] {
            let img = glyphs::rgba(d.line(i), d.meta.line_w, ROWS);
            if let Some(r) = g.set.read_all(&img, region) {
                let worst = r.glyphs.iter().map(|m| m.score).fold(f32::INFINITY, f32::min);
                let widths: Vec<u32> = r.glyphs.iter().map(|m| m.bounds.w).collect();
                println!("  {:?} read {:?}, weakest score {worst:.2}, glyph widths {widths:?}", d.truth[i], r.text);
            }
        }
    }
    }
}

/// Each line read `warmup` times, then `calls` times timed one by one; the
/// line's time is the median of its timed calls. Prints one line per test
/// line: kind and nanoseconds. With `pre`, the line is normalised before the
/// clock starts (as ONNX Runtime is handed it), else the time includes it.
fn time(dir: &Path, which: &str, warmup: usize, calls: usize, pre: bool) {
    let d = load(dir);
    let mut x = vec![0f32; H * W * 3];
    if which == "syrup" {
        return time_glyphs(&d, dir, warmup, calls);
    }
    let mut teacher = (which == "teacher").then(|| Teacher::load(dir, &d.meta));
    let mut reader = (which != "teacher").then(|| Reader::load(dir, &d.meta, which));
    let mut sink = 0usize;
    let mut one = |i: usize, x: &mut Vec<f32>| {
        if !pre {
            normalise(d.line(i), d.meta.line_w, x);
        }
        match (&mut teacher, &mut reader) {
            (Some(t), _) => t.read(x).len(),
            (_, Some(r)) => r.read(x).0.len(),
            _ => unreachable!(),
        }
    };
    let mut out = String::new();
    for i in 0..d.meta.lines {
        if pre {
            normalise(d.line(i), d.meta.line_w, &mut x);
        }
        for _ in 0..warmup {
            sink += one(i, &mut x);
        }
        let mut ns = vec![];
        for _ in 0..calls {
            let t0 = Instant::now();
            sink += one(i, &mut x);
            ns.push(t0.elapsed().as_nanos() as u64);
        }
        ns.sort_unstable();
        out.push_str(&format!("{} {}\n", d.kind[i], ns[ns.len() / 2]));
    }
    print!("{out}");
    eprintln!("({sink})");
}

/// `time` for Syrup's glyph reader: the font learned first, each line made
/// the RGBA image MapleSyrup would hand it before the clock starts.
fn time_glyphs(d: &Data, dir: &Path, warmup: usize, calls: usize) {
    let g = d.glyphs(dir, glyphs::MIN_SCORE);
    let region = glyphs::whole(d.meta.line_w, ROWS);
    let mut sink = 0usize;
    let mut out = String::new();
    for i in 0..d.meta.lines {
        let img = glyphs::rgba(d.line(i), d.meta.line_w, ROWS);
        for _ in 0..warmup {
            sink += g.read(&img, region).map_or(0, |t| t.len());
        }
        let mut ns = vec![];
        for _ in 0..calls {
            let t0 = Instant::now();
            sink += g.read(&img, region).map_or(0, |t| t.len());
            ns.push(t0.elapsed().as_nanos() as u64);
        }
        ns.sort_unstable();
        out.push_str(&format!("{} {}\n", d.kind[i], ns[ns.len() / 2]));
    }
    print!("{out}");
    eprintln!("({sink})");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("check") => check(Path::new(&args[2])),
        Some("time") => {
            let warmup = args.get(4).map_or(3, |s| s.parse().unwrap());
            let calls = args.get(5).map_or(5, |s| s.parse().unwrap());
            let pre = args.get(6).is_some_and(|s| s == "pre");
            time(Path::new(&args[2]), &args[3], warmup, calls, pre)
        }
        _ => eprintln!("usage: extract-port check DIR | time DIR teacher|white-box|black-box|syrup [WARMUP CALLS [pre]]"),
    }
}
