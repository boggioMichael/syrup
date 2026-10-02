use std::path::Path;
use std::process::ExitCode;

use image::Rgba;
use syrup_runtime::frames::{FrameSource, ImageFiles, WindowCapture};
use syrup_runtime::{
    FindResult, OwnedImage, PixelRect, RunParams, Runtime, SessionOptions, SyrupError,
};

const USAGE: &str = "usage:
  syrup explain <operation>
  syrup source <operation>
  syrup prepare <operation>...
  syrup bundle <dir> <operation>...
  syrup run <operation> <image> [--min-confidence F] [--max-results N] [--region X,Y,W,H] [--draw OUT]
  syrup watch <track_operation> (--window TITLE [--frames N] | <image>...)
  syrup windows
  syrup cache [path|list|clear]";

enum Failure {
    Usage(String),
    Syrup(SyrupError),
}

impl From<SyrupError> for Failure {
    fn from(e: SyrupError) -> Self {
        Failure::Syrup(e)
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(Failure::Usage(why)) => {
            eprintln!("{why}\n{USAGE}");
            ExitCode::from(2)
        }
        Err(Failure::Syrup(e)) => {
            eprintln!("error: {e}");
            for (key, value) in &e.details {
                eprintln!("  {key}: {value}");
            }
            ExitCode::FAILURE
        }
    }
}

fn run(args: &[String]) -> Result<(), Failure> {
    let usage = |why: &str| Failure::Usage(why.to_string());
    let runtime = Runtime::from_env()?;
    match args.first().map(String::as_str) {
        Some("explain") => print!("{}", runtime.resolve(arg(args, 1)?)?.explain()),
        Some("source") => print!("{}", runtime.resolve(arg(args, 1)?)?.source()),
        Some("prepare") => {
            if args.len() < 2 {
                return Err(usage("prepare needs at least one operation"));
            }
            for name in &args[1..] {
                let prepared = runtime.resolve(name)?.prepare()?;
                println!(
                    "{name}: {:?} in {:.0} ms -> {}",
                    prepared.status,
                    prepared.elapsed.as_secs_f64() * 1000.0,
                    prepared.library.display()
                );
            }
        }
        Some("bundle") => {
            if args.len() < 3 {
                return Err(usage("bundle needs a directory and at least one operation"));
            }
            let names: Vec<&str> = args[2..].iter().map(String::as_str).collect();
            let index = runtime.bundle(Path::new(&args[1]), &names)?;
            println!(
                "{}: {} operations for {}",
                args[1],
                index.operations.len(),
                index.target
            );
        }
        Some("run") => {
            let op = runtime.resolve(arg(args, 1)?)?;
            let image = OwnedImage::open(Path::new(arg(args, 2)?))?;
            let mut params = RunParams::default();
            let mut draw = None;
            let mut rest = args[3..].iter();
            while let Some(flag) = rest.next() {
                let value = rest
                    .next()
                    .ok_or_else(|| usage(&format!("{flag} needs a value")))?;
                let bad = || usage(&format!("bad value for {flag}: {value}"));
                match flag.as_str() {
                    "--min-confidence" => {
                        params.min_confidence = Some(value.parse().map_err(|_| bad())?)
                    }
                    "--max-results" => params.max_results = Some(value.parse().map_err(|_| bad())?),
                    "--region" => {
                        let n: Vec<u32> = value
                            .split(',')
                            .map(str::parse)
                            .collect::<Result<_, _>>()
                            .map_err(|_| bad())?;
                        let [x, y, w, h] = n[..] else {
                            return Err(bad());
                        };
                        params.region = Some(PixelRect { x, y, w, h });
                    }
                    "--draw" => draw = Some(value.clone()),
                    _ => return Err(usage(&format!("unknown option {flag}"))),
                }
            }
            let result = op.run(&image.as_input()?, &params)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&result).expect("results serialize")
            );
            if let Some(out) = draw {
                annotate(&image, &result)
                    .save(&out)
                    .map_err(|e| usage(&format!("cannot write {out}: {e}")))?;
            }
        }
        Some("watch") => {
            let op = runtime.resolve(arg(args, 1)?)?;
            let mut session = op.session(SessionOptions::default())?;
            let rest = &args[2..];
            let (mut source, limit): (Box<dyn FrameSource>, Option<u64>) = match rest {
                [flag, title, more @ ..] if flag == "--window" => {
                    let limit = match more {
                        [] => None,
                        [flag, n] if flag == "--frames" => Some(
                            n.parse()
                                .map_err(|_| usage(&format!("bad value for --frames: {n}")))?,
                        ),
                        _ => return Err(usage("watch --window TITLE takes only --frames N")),
                    };
                    (Box::new(WindowCapture::new(title)?), limit)
                }
                [] => return Err(usage("watch needs --window TITLE or image files")),
                files => (
                    Box::new(ImageFiles::new(files.iter().map(Into::into))),
                    None,
                ),
            };
            let mut seen = 0;
            while limit.is_none_or(|limit| seen < limit) {
                let Some(result) = session.next(source.as_mut(), &RunParams::default())? else {
                    break;
                };
                println!(
                    "{}",
                    serde_json::to_string(&result).expect("results serialize")
                );
                seen += 1;
            }
        }
        Some("windows") => {
            for title in WindowCapture::windows()? {
                println!("{title}");
            }
        }
        Some("cache") => match args.get(1).map(String::as_str) {
            None | Some("path") => println!("{}", runtime.store().root().display()),
            Some("list") => {
                for m in runtime.store().list() {
                    println!(
                        "{}  {}  ({} validation cases, {} ms)",
                        &m.key[..16],
                        m.intent,
                        m.validation_cases,
                        m.build_ms
                    );
                }
            }
            Some("clear") => runtime.store().clear()?,
            Some(other) => return Err(usage(&format!("unknown cache command {other}"))),
        },
        Some(other) => return Err(usage(&format!("unknown command {other}"))),
        None => return Err(usage("no command")),
    }
    Ok(())
}

fn arg(args: &[String], i: usize) -> Result<&str, Failure> {
    args.get(i)
        .map(String::as_str)
        .ok_or_else(|| Failure::Usage(format!("{} needs more arguments", args[0])))
}

fn annotate(image: &OwnedImage, result: &FindResult) -> image::RgbaImage {
    let mut out = image.to_rgba();
    let (green, yellow) = (Rgba([40, 220, 90, 255]), Rgba([255, 210, 0, 255]));
    for found in &result.items {
        let b = found.bbox;
        let (x, y) = (b.x.round() as u32, b.y.round() as u32);
        syrup::draw::draw_rect(
            &mut out,
            x,
            y,
            b.w.round() as u32,
            b.h.round() as u32,
            green,
            2,
        );
        let what = found.text.as_deref().unwrap_or(found.label);
        let label = format!("{what} {:.2}", found.confidence);
        let (width, height) = (out.width() as i64, out.height() as i64);
        syrup::draw::draw_text(&label, x as i64, y as i64 - 10, 1, |px, py| {
            if (0..width).contains(&px) && (0..height).contains(&py) {
                out.put_pixel(px as u32, py as u32, green);
            }
        });
        for k in &found.keypoints {
            let (kx, ky) = (k.x as i64, k.y as i64);
            for (dx, dy) in (-2..=2).flat_map(|d| [(d, 0), (0, d)]) {
                if (0..width).contains(&(kx + dx)) && (0..height).contains(&(ky + dy)) {
                    out.put_pixel((kx + dx) as u32, (ky + dy) as u32, yellow);
                }
            }
        }
    }
    out
}
