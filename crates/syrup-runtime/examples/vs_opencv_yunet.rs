//! F2 of `bench/vs-opencv` (see its PREREGISTRATION.md and AMENDMENTS.md):
//! Syrup's YuNet provider (the 2023mar ONNX model through tract) over the
//! prepared 640-pixel face images, with the same command line and JSON as
//! `examples/vs_opencv.rs` and `bench/vs-opencv/opencv_side.py yunet`.
//!
//! ```text
//! cargo run --release -p syrup-runtime --example vs_opencv_yunet -- \
//!     --data DIR --split dev|test [--calls K] [--warmup W] --out FILE.json
//! ```

use std::fmt::Write as _;
use std::path::PathBuf;
use std::time::Instant;

use syrup_runtime::providers::yunet::YuNet;
use syrup_runtime::providers::{ModelSource, Provider, ViewRef};

/// The score at which a candidate counts as a face (OpenCV's
/// `FaceDetectorYN` score threshold in the comparison).
const SCORE: f32 = 0.6;

fn main() {
    let mut args = std::env::args().skip(1);
    let (mut data, mut split, mut calls, mut warmup, mut out) = (
        PathBuf::new(),
        String::from("dev"),
        5usize,
        3usize,
        PathBuf::from("out.json"),
    );
    while let Some(flag) = args.next() {
        let value = args.next().expect("a value after each flag");
        match flag.as_str() {
            "--data" => data = value.into(),
            "--split" => split = value,
            "--calls" => calls = value.parse().expect("a number"),
            "--warmup" => warmup = value.parse().expect("a number"),
            "--out" => out = value.into(),
            "--threads" => {} // tract runs on the calling thread
            _ => panic!("unknown flag {flag}"),
        }
    }
    let started = Instant::now();
    let dir = data.join("faces640").join(&split);
    let mut inputs: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("the prepared directory")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "png"))
        .collect();
    inputs.sort();
    let load = Instant::now();
    let yunet = YuNet::new(ModelSource::Bundled);
    let load_ns = load.elapsed().as_nanos();
    let mut first_call_ns = None;
    let mut items = String::new();
    for (n, path) in inputs.iter().enumerate() {
        let image = image::open(path).expect("a decodable image").to_rgba8();
        let view = ViewRef {
            data: image.as_raw(),
            width: image.width(),
            height: image.height(),
            stride: image.width() as usize * 4,
            channels: 4,
        };
        let mut found = None;
        let mut first_of_input = None;
        for _ in 0..warmup {
            let t = Instant::now();
            found = Some(yunet.detect(&view).expect("YuNet runs"));
            first_of_input.get_or_insert(t.elapsed().as_nanos());
        }
        if first_call_ns.is_none() {
            first_call_ns = first_of_input;
        }
        let mut times = Vec::new();
        for _ in 0..calls {
            let t = Instant::now();
            found = Some(yunet.detect(&view).expect("YuNet runs"));
            times.push(t.elapsed().as_nanos().to_string());
        }
        let boxes: Vec<String> = found
            .expect("at least one call")
            .boxes
            .iter()
            .filter(|d| d.score >= SCORE)
            .map(|d| {
                let points: Vec<String> = d.keypoints[..10]
                    .iter()
                    .map(|v| format!("{v:.2}"))
                    .collect();
                format!(
                    "[{:.2},{:.2},{:.2},{:.2},{:.4},{}]",
                    d.x,
                    d.y,
                    d.w,
                    d.h,
                    d.score,
                    points.join(",")
                )
            })
            .collect();
        if n > 0 {
            items.push(',');
        }
        let input = path.strip_prefix(&data).unwrap_or(path).to_string_lossy();
        let _ = write!(
            items,
            "{{\"input\":\"{input}\",\"times_ns\":[{}],\"init_first_call_ns\":{},\"output\":{{\"boxes\":[{}]}}}}",
            times.join(","),
            first_of_input.unwrap_or(0),
            boxes.join(",")
        );
    }
    let wall = started.elapsed().as_secs_f64();
    let stat = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let fields: Vec<&str> = stat
        .rsplit_once(')')
        .map(|(_, r)| r)
        .unwrap_or("")
        .split_whitespace()
        .collect();
    let ticks = |i: usize| {
        fields
            .get(i)
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.0)
            / 100.0
    };
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let rss: u64 = status
        .lines()
        .find(|l| l.starts_with("VmHWM:"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let json = format!(
        "{{\"impl\":\"syrup\",\"version\":\"{}\",\"task\":\"yunet\",\"split\":\"{split}\",\"threads\":1,\"calls\":{calls},\"warmup\":{warmup},\"init\":{{\"load_ns\":{load_ns},\"first_call_ns\":{}}},\"process\":{{\"wall_s\":{wall},\"cpu_user_s\":{},\"cpu_sys_s\":{},\"max_rss_kb\":{rss}}},\"items\":[{items}]}}\n",
        option_env!("SYRUP_BENCH_VERSION").unwrap_or("unknown"),
        first_call_ns.unwrap_or(0),
        ticks(11),
        ticks(12)
    );
    std::fs::write(&out, json).expect("the output written");
}
