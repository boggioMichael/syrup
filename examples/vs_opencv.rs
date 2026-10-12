//! The Syrup side of `bench/vs-opencv` (see its PREREGISTRATION.md): runs one
//! task over the prepared inputs and writes per-call timings, outputs and
//! the process's CPU time and peak memory as JSON. Its twin,
//! `bench/vs-opencv/opencv_side.py`, does the same with OpenCV.
//!
//! ```text
//! cargo run --release --example vs_opencv -- TASK --data DIR --split dev|test \
//!     [--threads N] [--calls K] [--warmup W] [--dump DIR] [--min-deviation D] \
//!     [--min-size S] [--min-neighbors N] \
//!     --out FILE.json
//! ```
//!
//! Tasks: `faces` (Haar cascade on the prepared grey images), `templates`
//! (best match of each prepared template), `motion` (moved-pixel mask, and
//! the changed regions, of consecutive frames). Each image input is called
//! `warmup` times untimed, then `calls` times timed; a video is processed
//! once, front to back, its first `warmup` pairs untimed.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use syrup::cascade::{Cascade, CascadeOptions};
use syrup::geometry::Rect;
use syrup::glyphs::{GlyphOptions, GlyphSet};
use syrup::image::{self, GrayImage, RgbaImage};
use syrup::motion::{self, MotionConfig};
use syrup::template::{self, Template};
use syrup::threshold::Channel;

struct Args {
    task: String,
    data: PathBuf,
    split: String,
    threads: usize,
    calls: usize,
    warmup: usize,
    dump: Option<PathBuf>,
    out: PathBuf,
    /// `faces`: `CascadeOptions::min_deviation` (OpenCV's window gate is 10).
    min_deviation: Option<f32>,
    /// `faces`: smallest face side (0: the cascade's 20-pixel window) and
    /// minimum neighbours; the comparison uses 0 and 3.
    min_size: u32,
    min_neighbors: u32,
}

fn args() -> Args {
    let mut it = std::env::args().skip(1);
    let task = it.next().unwrap_or_else(|| usage());
    let mut a = Args {
        task,
        data: PathBuf::new(),
        split: "dev".into(),
        threads: 1,
        calls: 5,
        warmup: 3,
        dump: None,
        out: PathBuf::from("out.json"),
        min_deviation: None,
        min_size: 0,
        min_neighbors: 3,
    };
    while let Some(flag) = it.next() {
        let mut value = || it.next().unwrap_or_else(|| usage());
        match flag.as_str() {
            "--data" => a.data = value().into(),
            "--split" => a.split = value(),
            "--threads" => a.threads = value().parse().unwrap_or_else(|_| usage()),
            "--calls" => a.calls = value().parse().unwrap_or_else(|_| usage()),
            "--warmup" => a.warmup = value().parse().unwrap_or_else(|_| usage()),
            "--dump" => a.dump = Some(value().into()),
            "--out" => a.out = value().into(),
            "--min-deviation" => {
                a.min_deviation = Some(value().parse().unwrap_or_else(|_| usage()))
            }
            "--min-size" => a.min_size = value().parse().unwrap_or_else(|_| usage()),
            "--min-neighbors" => a.min_neighbors = value().parse().unwrap_or_else(|_| usage()),
            _ => usage(),
        }
    }
    a
}

fn usage() -> ! {
    eprintln!(
        "usage: vs_opencv faces|templates|motion|glyphs --data DIR --split dev|test [--threads N] [--calls K] [--warmup W] [--dump DIR] [--min-deviation D] [--min-size S] [--min-neighbors N] --out FILE"
    );
    std::process::exit(2)
}

/// One input's timed calls and what the last call gave.
struct Item {
    input: String,
    times_ns: Vec<u128>,
    /// Extra timings (template preparation).
    extra_ns: Vec<(String, u128)>,
    output: String,
}

fn time<T>(f: impl FnOnce() -> T) -> (T, u128) {
    let start = Instant::now();
    let value = f();
    (value, start.elapsed().as_nanos())
}

fn rect_json(r: &Rect) -> String {
    format!("[{},{},{},{}]", r.x, r.y, r.w, r.h)
}

fn sorted_pngs(dir: &Path, keep: impl Fn(&str) -> bool) -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.extension().is_some_and(|x| x == "png")
                && keep(&p.file_name().unwrap().to_string_lossy())
        })
        .collect();
    found.sort();
    found
}

fn relative(path: &Path, data: &Path) -> String {
    path.strip_prefix(data)
        .unwrap_or(path)
        .to_string_lossy()
        .into_owned()
}

fn grey(path: &Path) -> GrayImage {
    image::open(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .to_luma8()
}

fn rgba(path: &Path) -> RgbaImage {
    image::open(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .to_rgba8()
}

/// A grey image as the RGBA the template API takes (Syrup's luma maps a
/// grey to itself, so the searched values are the same bytes).
fn grey_rgba(grey: &GrayImage) -> RgbaImage {
    RgbaImage::from_fn(grey.width(), grey.height(), |x, y| {
        let v = grey.get_pixel(x, y).0[0];
        image::Rgba([v, v, v, 255])
    })
}

fn faces(a: &Args) -> (Vec<Item>, String) {
    let dir = a.data.join("faces").join(&a.split);
    let inputs = sorted_pngs(&dir, |n| n.ends_with(".grey.png"));
    let (cascade, load_ns) = time(Cascade::frontal_face);
    let options = CascadeOptions {
        scale_factor: 1.1,
        min_neighbors: a.min_neighbors,
        min_size: a.min_size,
        max_size: None,
        threads: a.threads,
        min_deviation: a.min_deviation,
    };
    let mut items = Vec::new();
    let mut first_call_ns = None;
    for path in &inputs {
        let image = grey(path);
        let mut found = Vec::new();
        for i in 0..a.warmup {
            let (f, t) = time(|| cascade.detect(&image, &options));
            if i == 0 && first_call_ns.is_none() {
                first_call_ns = Some(t);
            }
            found = f;
        }
        let mut times = Vec::new();
        for _ in 0..a.calls {
            let (f, t) = time(|| cascade.detect(&image, &options));
            times.push(t);
            found = f;
        }
        let boxes: Vec<String> = found
            .iter()
            .map(|m| {
                format!(
                    "[{},{},{},{},{}]",
                    m.bounds.x, m.bounds.y, m.bounds.w, m.bounds.h, m.neighbors
                )
            })
            .collect();
        items.push(Item {
            input: relative(path, &a.data),
            times_ns: times,
            extra_ns: Vec::new(),
            output: format!("{{\"boxes\":[{}]}}", boxes.join(",")),
        });
    }
    let init = format!(
        "{{\"load_ns\":{load_ns},\"first_call_ns\":{}}}",
        first_call_ns.unwrap_or(0)
    );
    (items, init)
}

fn templates(a: &Args) -> (Vec<Item>, String) {
    let dir = a.data.join("templates").join(&a.split);
    let cases: Vec<(String, String, String)> = parse_cases(&dir.join("cases.json"));
    let mut images: std::collections::HashMap<String, RgbaImage> = Default::default();
    let mut items = Vec::new();
    let mut first_call_ns = None;
    for (image_rel, template_rel, label) in &cases {
        let image = images
            .entry(image_rel.clone())
            .or_insert_with(|| grey_rgba(&grey(&a.data.join(image_rel))));
        let template_rgba = grey_rgba(&grey(&a.data.join(template_rel)));
        let (template, prep_ns) = time(|| Template::from_image(&template_rgba, Channel::Luma));
        let Some(template) = template else {
            items.push(Item {
                input: label.clone(),
                times_ns: Vec::new(),
                extra_ns: vec![("prep_ns".into(), prep_ns)],
                output: "{\"refused\":true}".into(),
            });
            continue;
        };
        let search = Rect {
            x: 0,
            y: 0,
            w: image.width(),
            h: image.height(),
        };
        let mut best = None;
        for i in 0..a.warmup {
            let (b, t) = time(|| template::find_best(image, search, &template));
            if i == 0 && first_call_ns.is_none() {
                first_call_ns = Some(t);
            }
            best = b;
        }
        let mut times = Vec::new();
        for _ in 0..a.calls {
            let (b, t) = time(|| template::find_best(image, search, &template));
            times.push(t);
            best = b;
        }
        let output = match best {
            Some(m) => format!(
                "{{\"x\":{},\"y\":{},\"score\":{}}}",
                m.bounds.x, m.bounds.y, m.score
            ),
            None => "{\"none\":true}".into(),
        };
        items.push(Item {
            input: label.clone(),
            times_ns: times,
            extra_ns: vec![("prep_ns".into(), prep_ns)],
            output,
        });
    }
    (
        items,
        format!("{{\"first_call_ns\":{}}}", first_call_ns.unwrap_or(0)),
    )
}

/// The `(image, template, label)` of every case in `cases.json`: a label
/// that names the case for both sides (its index, image, template, variant).
fn parse_cases(path: &Path) -> Vec<(String, String, String)> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    // A tiny reader for this file's shape: a list of flat objects.
    let mut cases = Vec::new();
    for (index, object) in text.split('{').skip(1).enumerate() {
        let field = |name: &str| -> String {
            let key = format!("\"{name}\":");
            let at = object
                .find(&key)
                .unwrap_or_else(|| panic!("case {index}: no {name}"))
                + key.len();
            let rest = object[at..].trim_start();
            if let Some(stripped) = rest.strip_prefix('"') {
                stripped[..stripped.find('"').unwrap()].to_string()
            } else {
                rest.split([',', '}']).next().unwrap().trim().to_string()
            }
        };
        let (image, template, variant) = (field("image"), field("template"), field("variant"));
        let label = format!("{index}|{template}|{variant}");
        cases.push((image, template, label));
    }
    cases
}

fn motion_task(a: &Args) -> (Vec<Item>, String) {
    let root = a.data.join("motion").join(&a.split);
    let mut sources: Vec<PathBuf> = std::fs::read_dir(&root)
        .unwrap_or_else(|e| panic!("{}: {e}", root.display()))
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    sources.sort();
    let config = MotionConfig::default();
    let mut items = Vec::new();
    for source in &sources {
        let frames: Vec<RgbaImage> = sorted_pngs(source, |_| true)
            .iter()
            .map(|p| rgba(p))
            .collect();
        let name = source.file_name().unwrap().to_string_lossy().into_owned();
        for (i, pair) in frames.windows(2).enumerate() {
            // The timed call first, so it does not find the frames already
            // in cache from the mask computed for the comparison.
            let (regions, regions_ns) = time(|| motion::regions(&pair[0], &pair[1], &config));
            let (mask, mask_ns) =
                time(|| motion::motion_mask(&pair[0], &pair[1], config.diff_threshold));
            let mask = mask.expect("same size");
            let regions = regions.expect("same size");
            if let Some(dump) = &a.dump {
                let dir = dump.join(&name);
                std::fs::create_dir_all(&dir).ok();
                mask.save(dir.join(format!("{i:04}.png")))
                    .expect("mask saved");
            }
            if i < a.warmup {
                continue;
            }
            let moved = mask.as_raw().iter().filter(|&&v| v != 0).count();
            let boxes: Vec<String> = regions.iter().map(rect_json).collect();
            items.push(Item {
                input: format!("{name}/{i:04}"),
                times_ns: vec![regions_ns],
                extra_ns: vec![("mask_ns".into(), mask_ns)],
                output: format!("{{\"moved\":{moved},\"regions\":[{}]}}", boxes.join(",")),
            });
        }
    }
    (items, "{}".into())
}

/// Lines of tab-separated fields.
fn tsv(path: &Path) -> Vec<Vec<String>> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| l.split('\t').map(str::to_string).collect())
        .collect()
}

fn json_string(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// F5: one glyph set learns the font from each size's labelled line
/// (default options: light text), then reads every case.
fn glyphs_task(a: &Args) -> (Vec<Item>, String) {
    let dir = a.data.join("glyphs").join(&a.split);
    let mut set = GlyphSet::new(GlyphOptions::default());
    let mut learned = Vec::new();
    let (_, learn_ns) = time(|| {
        for row in tsv(&dir.join("train.tsv")) {
            let image = rgba(&a.data.join(&row[0]));
            let whole = Rect {
                x: 0,
                y: 0,
                w: image.width(),
                h: image.height(),
            };
            learned.push(match set.learn(&image, whole, &row[1]) {
                Ok(n) => format!("{{\"file\":{},\"learned\":{n}}}", json_string(&row[0])),
                Err(e) => format!(
                    "{{\"file\":{},\"error\":{}}}",
                    json_string(&row[0]),
                    json_string(&e.to_string())
                ),
            });
        }
    });
    let mut items = Vec::new();
    let mut first_call_ns = None;
    for row in tsv(&dir.join("cases.tsv")) {
        let image = rgba(&a.data.join(&row[0]));
        let whole = Rect {
            x: 0,
            y: 0,
            w: image.width(),
            h: image.height(),
        };
        let mut reading = None;
        for i in 0..a.warmup {
            let (r, t) = time(|| set.read(&image, whole));
            if i == 0 && first_call_ns.is_none() {
                first_call_ns = Some(t);
            }
            reading = Some(r);
        }
        let mut times = Vec::new();
        for _ in 0..a.calls {
            let (r, t) = time(|| set.read(&image, whole));
            times.push(t);
            reading = Some(r);
        }
        let reading = reading.expect("at least one call");
        let output = match &reading.value {
            Some(text) => format!("{{\"text\":{}}}", json_string(&text.text)),
            None => format!(
                "{{\"refused\":{}}}",
                json_string(reading.failure_reason.as_deref().unwrap_or(""))
            ),
        };
        items.push(Item {
            input: row[0].clone(),
            times_ns: times,
            extra_ns: Vec::new(),
            output,
        });
    }
    let init = format!(
        "{{\"load_ns\":{learn_ns},\"first_call_ns\":{},\"learned\":[{}]}}",
        first_call_ns.unwrap_or(0),
        learned.join(",")
    );
    (items, init)
}

/// This process's CPU time (user, system, seconds) and peak resident set
/// (KiB), from /proc (Linux).
fn process_usage() -> (f64, f64, u64) {
    let ticks = 100.0; // USER_HZ on Linux
    let stat = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let after = stat.rsplit_once(')').map(|(_, r)| r).unwrap_or("");
    let fields: Vec<&str> = after.split_whitespace().collect();
    let utime = fields
        .get(11)
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.0)
        / ticks;
    let stime = fields
        .get(12)
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.0)
        / ticks;
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let hwm = status
        .lines()
        .find(|l| l.starts_with("VmHWM:"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    (utime, stime, hwm)
}

fn main() {
    let a = args();
    let started = Instant::now();
    let (items, init) = match a.task.as_str() {
        "faces" => faces(&a),
        "templates" => templates(&a),
        "motion" => motion_task(&a),
        "glyphs" => glyphs_task(&a),
        _ => usage(),
    };
    let wall = started.elapsed().as_secs_f64();
    let (user, system, rss) = process_usage();
    let mut out = String::new();
    let _ = write!(
        out,
        "{{\"impl\":\"syrup\",\"version\":\"{}\",\"task\":\"{}\",\"split\":\"{}\",\"threads\":{},\"calls\":{},\"warmup\":{},\"init\":{init},",
        option_env!("SYRUP_BENCH_VERSION").unwrap_or("unknown"),
        a.task,
        a.split,
        a.threads,
        a.calls,
        a.warmup
    );
    let _ = write!(
        out,
        "\"process\":{{\"wall_s\":{wall},\"cpu_user_s\":{user},\"cpu_sys_s\":{system},\"max_rss_kb\":{rss}}},\"items\":["
    );
    for (n, item) in items.iter().enumerate() {
        if n > 0 {
            out.push(',');
        }
        let times: Vec<String> = item.times_ns.iter().map(|t| t.to_string()).collect();
        let extra: Vec<String> = item
            .extra_ns
            .iter()
            .map(|(k, v)| format!("\"{k}\":{v}"))
            .collect();
        let _ = write!(
            out,
            "{{\"input\":\"{}\",\"times_ns\":[{}],{}\"output\":{}}}",
            item.input,
            times.join(","),
            if extra.is_empty() {
                String::new()
            } else {
                format!("{},", extra.join(","))
            },
            item.output
        );
    }
    out.push_str("]}\n");
    std::fs::write(&a.out, out).unwrap_or_else(|e| panic!("{}: {e}", a.out.display()));
}
