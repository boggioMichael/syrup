//! Text recognition for small on-screen UI text.
//!
//! Two engines: a Tesseract subprocess, located via `TESSERACT_BIN`, `PATH`,
//! or the standard Windows install locations, and on Windows the OS's
//! built-in OCR engine ([`windows`]), which is trained on screen content
//! and often beats Tesseract on pixel-font UI text. [`engine`] says which
//! one [`ocr_region`] uses on this machine: Tesseract where it is
//! installed, else the Windows engine, else none; `SYRUP_OCR=tesseract` or
//! `SYRUP_OCR=windows` chooses explicitly.
//!
//! For text drawn in one fixed pixel font — counters, HUD values — the
//! template reader in [`crate::glyphs`] is faster and more reliable than
//! either engine, because it knows the exact font.
//!
//! [`recognize`] says why recognition did not run ([`OcrError`]) and gives
//! word boxes in the coordinates of the image it was handed; [`ocr_region`]
//! and [`ocr_region_with`] answer `None` for every failure and for no text,
//! with word boxes relative to the crop.

pub mod windows;

use std::env;
use std::ffi::OsString;
use std::fmt;
use std::fs::{self, OpenOptions};
use std::io::{BufWriter, ErrorKind};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use image::{DynamicImage, GrayImage, ImageOutputFormat, Luma, RgbaImage, imageops};

use crate::detection::Confidence;
use crate::geometry::Rect;

/// OCR configuration for a single crop.
#[derive(Debug, Clone)]
pub struct OcrConfig {
    /// Page segmentation mode passed to Tesseract.
    pub psm: u8,
    /// Optional whitelist of characters to prefer.
    pub whitelist: Option<String>,
}

impl Default for OcrConfig {
    fn default() -> Self {
        Self {
            // PSM 0 only performs orientation detection and never recognizes text.
            // Sparse text handles interfaces where labels and values are separated.
            psm: 11,
            whitelist: None,
        }
    }
}

impl OcrConfig {
    /// The crop is one line of text (page segmentation mode 7), such as a
    /// single label or value; more reliable than sparse-text mode when true.
    pub fn single_line() -> Self {
        Self {
            psm: 7,
            whitelist: None,
        }
    }

    /// Restrict recognition to these characters, e.g. `"0123456789/"` for a
    /// counter, so look-alike letters cannot be returned.
    pub fn with_whitelist(mut self, chars: impl Into<String>) -> Self {
        self.whitelist = Some(chars.into());
        self
    }

    /// Option flags only, without the leading positional arguments or the
    /// trailing config-file name.
    fn to_args(&self) -> Vec<String> {
        let mut args = vec![
            "--oem".to_string(),
            "3".to_string(),
            "--psm".to_string(),
            self.psm.to_string(),
        ];
        if let Some(whitelist) = &self.whitelist {
            args.push("-c".to_string());
            args.push(format!("tessedit_char_whitelist={whitelist}"));
        }

        args.push("-c".to_string());
        args.push("preserve_interword_spaces=1".to_string());
        args
    }
}

/// Build the full Tesseract command line for one crop.
///
/// Tesseract's grammar is `tesseract IMAGE OUTPUTBASE [options...] [configfile...]`
/// and its parser stops reading options at the first non-flag argument that
/// follows the two positional ones. Putting the `tsv` config file before
/// `--oem`/`--psm`/`-c` makes Tesseract treat every flag as a config file name
/// ("read_params_file: Can't open --oem") and silently fall back to its default
/// page segmentation mode, so `tsv` has to come last.
fn tesseract_args(input: &Path, config: &OcrConfig) -> Vec<OsString> {
    let mut args = vec![input.as_os_str().to_os_string(), OsString::from("stdout")];
    args.extend(config.to_args().into_iter().map(OsString::from));
    args.push(OsString::from("tsv"));
    args
}

/// Owns a temporary OCR input image and deletes it on drop.
///
/// The Tesseract call has several fallible steps after the PNG is written; an
/// early return from any of them used to leak the file into the temp
/// directory. Tying deletion to the value's lifetime covers every exit path.
struct TempImage {
    path: PathBuf,
}

impl TempImage {
    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempImage {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

/// OCR result for a single crop.
#[derive(Debug, Clone, Default)]
pub struct OcrResult {
    pub text: String,
    pub available: bool,
    pub words: Vec<OcrWord>,
}

impl OcrResult {
    /// How sure the engine is of the text: the mean of its per-word
    /// confidences, or none when no word carries one.
    pub fn confidence(&self) -> Confidence {
        if self.words.is_empty() {
            return Confidence::NONE;
        }
        let total: f32 = self.words.iter().map(|word| word.confidence).sum();
        Confidence::new(total / self.words.len() as f32)
    }
}

/// A word recognized by OCR and its bounding rectangle in crop coordinates.
#[derive(Debug, Clone)]
pub struct OcrWord {
    pub text: String,
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
    /// The engine's confidence in this word, in `[0, 1]`.
    pub confidence: f32,
}

/// Which recogniser reads text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Engine {
    /// Tesseract, an external program (`TESSERACT_BIN`, the PATH, or its
    /// standard Windows install folders).
    Tesseract,
    /// The OCR engine built into Windows, used when Tesseract is not
    /// installed (or when `SYRUP_OCR=windows` asks for it).
    Windows,
}

/// The recogniser [`ocr_region`] uses on this machine, if any.
///
/// Tesseract comes first where it is installed, since it gives word boxes
/// and confidences and behaves the same on every platform. Without it — the
/// usual case on a Windows PC — the engine Windows ships with reads the
/// text instead of nothing at all. `SYRUP_OCR=tesseract` or
/// `SYRUP_OCR=windows` chooses explicitly (when that engine is there).
pub fn engine() -> Option<Engine> {
    let tesseract = find_tesseract_binary().is_some();
    let windows = windows_ocr_available();
    match env::var("SYRUP_OCR").ok().as_deref() {
        Some("windows") if windows => Some(Engine::Windows),
        Some("tesseract") if tesseract => Some(Engine::Tesseract),
        _ if tesseract => Some(Engine::Tesseract),
        _ if windows => Some(Engine::Windows),
        _ => None,
    }
}

/// Whether the Windows engine can be created, asked once: it depends on an
/// installed language pack, which does not change while running.
fn windows_ocr_available() -> bool {
    static AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *AVAILABLE.get_or_init(windows::is_available)
}

/// Run OCR over an image crop with the default configuration and return
/// the recognized text. See [`ocr_region_with`].
pub fn ocr_region(image: &RgbaImage, x: u32, y: u32, w: u32, h: u32) -> Option<OcrResult> {
    ocr_region_with(image, x, y, w, h, &OcrConfig::default())
}

/// Run OCR over an image crop and return the recognized text, with the
/// engine [`engine`] picks.
///
/// The crop is clipped to the image. Returns `None` when no engine is
/// available, the crop is empty, or nothing was recognised. With Tesseract,
/// word boxes come back relative to the crop's top-left corner ([`recognize`]
/// says why a read failed instead); the Windows engine gives the text alone.
pub fn ocr_region_with(
    image: &RgbaImage,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    config: &OcrConfig,
) -> Option<OcrResult> {
    let engine = engine()?;
    if engine == Engine::Windows {
        let crop = crop_region(image, x, y, w, h)?;
        let prepared = prepare_for_windows(&crop);
        let text = normalize_text(&windows::recognize(&prepared)?);
        if text.trim().is_empty() {
            return None;
        }
        return Some(OcrResult {
            text,
            available: true,
            words: Vec::new(),
        });
    }
    let binary = find_tesseract_binary()?;
    let words = run_tesseract(&binary, image, Rect { x, y, w, h }, config).ok()?;
    let text = joined_text(words.iter().map(|word| word.text.as_str()));
    if text.is_empty() {
        return None;
    }

    Some(OcrResult {
        text,
        available: true,
        words,
    })
}

/// A recognised word, in the coordinates of the image given to [`recognize`].
#[derive(Debug, Clone, PartialEq)]
pub struct Word {
    pub text: String,
    pub bounds: Rect,
    /// Tesseract's confidence, scaled to `[0, 1]`.
    pub confidence: f32,
}

/// What [`recognize`] read: the words, and their text joined by spaces.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Recognition {
    /// The words joined by spaces and normalised.
    pub text: String,
    pub words: Vec<Word>,
}

/// Why recognition did not run. Recognising no text is not an error: it is
/// an empty [`Recognition`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OcrError {
    /// No Tesseract on `TESSERACT_BIN`, `PATH` or the standard install paths.
    EngineMissing,
    /// The region does not overlap the image.
    EmptyRegion,
    /// The temporary input image could not be written.
    Io(String),
    /// Tesseract ran and failed, or could not be started.
    EngineFailed { status: Option<i32>, stderr: String },
}

impl fmt::Display for OcrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OcrError::EngineMissing => {
                f.write_str("Tesseract is not installed (set TESSERACT_BIN or add it to PATH)")
            }
            OcrError::EmptyRegion => f.write_str("the region does not overlap the image"),
            OcrError::Io(e) => write!(f, "cannot write the OCR input image: {e}"),
            OcrError::EngineFailed { status, stderr } => match status {
                Some(code) => write!(f, "Tesseract exited with status {code}: {stderr}"),
                None => write!(f, "Tesseract could not run: {stderr}"),
            },
        }
    }
}

impl std::error::Error for OcrError {}

/// Recognise the text in `region` of `image`. Word boxes come back in
/// `image`'s coordinates, whatever preprocessing happened in between; no
/// text is an empty [`Recognition`], and anything that stops Tesseract from
/// reading is an [`OcrError`].
pub fn recognize(
    image: &RgbaImage,
    region: Rect,
    config: &OcrConfig,
) -> Result<Recognition, OcrError> {
    let binary = find_tesseract_binary().ok_or(OcrError::EngineMissing)?;
    recognize_with(&binary, image, region, config)
}

fn recognize_with(
    binary: &Path,
    image: &RgbaImage,
    region: Rect,
    config: &OcrConfig,
) -> Result<Recognition, OcrError> {
    let words: Vec<Word> = run_tesseract(binary, image, region, config)?
        .into_iter()
        .map(|word| Word {
            bounds: Rect {
                x: region.x + word.x,
                y: region.y + word.y,
                w: word.w,
                h: word.h,
            },
            text: word.text,
            confidence: word.confidence,
        })
        .collect();
    let text = joined_text(words.iter().map(|word| word.text.as_str()));
    Ok(Recognition { text, words })
}

fn joined_text<'a>(words: impl Iterator<Item = &'a str>) -> String {
    normalize_text(&words.collect::<Vec<_>>().join(" "))
}

/// One Tesseract run over `region` of `image`: the words it read, with
/// boxes relative to the crop (the region clipped to the image).
fn run_tesseract(
    binary: &Path,
    image: &RgbaImage,
    region: Rect,
    config: &OcrConfig,
) -> Result<Vec<OcrWord>, OcrError> {
    let crop =
        crop_region(image, region.x, region.y, region.w, region.h).ok_or(OcrError::EmptyRegion)?;
    let (width, height) = crop.dimensions();

    // Every call starts a process, so the crop is prepared once, the way
    // that measured best, rather than trying several variants.
    let (input_image, placement) = preprocess_image(&crop);
    // `input` deletes the PNG when it drops, on every path out of here.
    let input = write_temp_image(&input_image).ok_or_else(|| {
        OcrError::Io(format!(
            "no file could be created in {}",
            env::temp_dir().display()
        ))
    })?;
    let mut command = Command::new(binary);
    command.args(tesseract_args(input.path(), config));
    // Tesseract's OpenMP threads cost more to start than they save on a
    // crop this small; one thread is measurably faster. An explicit
    // setting in the environment still wins.
    if env::var_os("OMP_THREAD_LIMIT").is_none() {
        command.env("OMP_THREAD_LIMIT", "1");
    }

    let output = command.output().map_err(|e| OcrError::EngineFailed {
        status: None,
        stderr: e.to_string(),
    })?;
    if !output.status.success() {
        return Err(OcrError::EngineFailed {
            status: output.status.code(),
            stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
        });
    }
    Ok(parse_tsv_words(&String::from_utf8_lossy(&output.stdout))
        .into_iter()
        .map(|word| unscale_word(word, placement, width, height))
        .collect())
}

/// Words from Tesseract's TSV output, in reading order, with boxes in the
/// coordinates of the image Tesseract was given.
///
/// Columns: level, page, block, paragraph, line, word, left, top, width,
/// height, conf, text. Only word rows carry text; the page, block and line
/// rows above them are skipped.
fn parse_tsv_words(tsv: &str) -> Vec<OcrWord> {
    tsv.lines()
        .skip(1)
        .filter_map(|line| {
            let fields = line.split('\t').collect::<Vec<_>>();
            if fields.len() < 12 || fields[11].trim().is_empty() {
                return None;
            }
            let confidence = fields[10].trim().parse::<f32>().ok()?;
            Some(OcrWord {
                text: fields[11].trim().to_string(),
                x: fields[6].parse().ok()?,
                y: fields[7].parse().ok()?,
                w: fields[8].parse().ok()?,
                h: fields[9].parse().ok()?,
                confidence: (confidence / 100.0).clamp(0.0, 1.0),
            })
        })
        .collect()
}

/// Map a word box from the image Tesseract read back onto the crop: the
/// smallest box of crop pixels covering it, clipped to the crop.
fn unscale_word(mut word: OcrWord, placement: Placement, width: u32, height: u32) -> OcrWord {
    let scale = placement.scale.max(1);
    let to_crop_start =
        |v: u32, limit: u32| (v.saturating_sub(placement.margin) / scale).min(limit);
    let to_crop_end = |v: u32, limit: u32| {
        v.saturating_sub(placement.margin)
            .div_ceil(scale)
            .min(limit)
    };
    let x0 = to_crop_start(word.x, width);
    let y0 = to_crop_start(word.y, height);
    let x1 = to_crop_end(word.x.saturating_add(word.w), width);
    let y1 = to_crop_end(word.y.saturating_add(word.h), height);
    word.x = x0;
    word.y = y0;
    word.w = x1.saturating_sub(x0);
    word.h = y1.saturating_sub(y0);
    word
}

/// Check whether an OCR backend is available on the current machine.
pub fn is_ocr_available() -> bool {
    engine().is_some()
}

/// What the Windows engine reads best: the crop in grey, enlarged so the
/// text is tall enough, its contrast raised and its strokes sharpened.
fn prepare_for_windows(crop: &RgbaImage) -> RgbaImage {
    let (gray, _) = upscale_for_ocr(imageops::grayscale(crop));
    DynamicImage::ImageLuma8(gray)
        .adjust_contrast(45.0)
        .unsharpen(1.0, 1)
        .to_rgba8()
}

/// The part of `image` inside the rectangle, or `None` when that part is
/// empty. Safe for any rectangle, including ones reaching past `u32::MAX`.
fn crop_region(image: &RgbaImage, x: u32, y: u32, w: u32, h: u32) -> Option<RgbaImage> {
    let x_end = x.saturating_add(w).min(image.width());
    let y_end = y.saturating_add(h).min(image.height());
    if x_end <= x || y_end <= y {
        return None;
    }
    Some(imageops::crop_imm(image, x, y, x_end - x, y_end - y).to_image())
}

/// Crop height, in pixels, that OCR is given to work with.
///
/// On-screen UI text is often drawn 8-10 pixels tall, far below what
/// Tesseract is trained for, and at that size it returns near-noise.
/// Enlarging the crop first is what makes small UI text legible to it, so
/// crops are scaled up to roughly this height.
const MIN_OCR_TEXT_HEIGHT: u32 = 48;

/// Smallest white margin, in pixels, framing the image Tesseract reads.
const MIN_OCR_MARGIN: u32 = 10;

/// Where the crop sits inside the image Tesseract read: enlarged by
/// `scale`, then framed by `margin` pixels on every side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Placement {
    scale: u32,
    margin: u32,
}

/// Turn a crop into what Tesseract is trained on: dark text on a light,
/// evenly lit page with a margin around it.
///
/// 1. Grayscale, inverted when the crop's border — its background — is
///    dark, so light-on-dark UI text becomes dark-on-light.
/// 2. Levels stretched so background and ink span the full range, which
///    also lifts low-contrast text.
/// 3. Enlarged so the text is tall enough to read.
/// 4. Framed with a white margin: text that touches the image edge is
///    routinely dropped whole in sparse-text mode.
fn preprocess_image(image: &RgbaImage) -> (DynamicImage, Placement) {
    let mut gray = imageops::grayscale(image);
    if border_median(&gray).is_some_and(|background| background < 128) {
        imageops::invert(&mut gray);
    }
    stretch_levels(&mut gray);
    let (gray, scale) = upscale_for_ocr(gray);
    let margin = (gray.height() / 2).max(MIN_OCR_MARGIN);
    let mut framed = GrayImage::from_pixel(
        gray.width() + 2 * margin,
        gray.height() + 2 * margin,
        Luma([255]),
    );
    imageops::replace(&mut framed, &gray, i64::from(margin), i64::from(margin));
    (
        DynamicImage::ImageLuma8(framed),
        Placement { scale, margin },
    )
}

/// Median of the outermost ring of pixels: the background, for a crop
/// drawn around text. `None` for an empty image.
fn border_median(gray: &GrayImage) -> Option<u8> {
    let (width, height) = gray.dimensions();
    if width == 0 || height == 0 {
        return None;
    }
    let mut ring: Vec<u8> = Vec::with_capacity(2 * (width + height) as usize);
    for x in 0..width {
        ring.push(gray.get_pixel(x, 0)[0]);
        ring.push(gray.get_pixel(x, height - 1)[0]);
    }
    for y in 0..height {
        ring.push(gray.get_pixel(0, y)[0]);
        ring.push(gray.get_pixel(width - 1, y)[0]);
    }
    let middle = ring.len() / 2;
    Some(*ring.select_nth_unstable(middle).1)
}

/// Linearly stretch the 2nd-98th percentile range of `gray` to 0-255.
/// Near-uniform images are left alone: there is no text to bring out, and
/// stretching would only amplify noise.
fn stretch_levels(gray: &mut GrayImage) {
    let histogram = crate::threshold::histogram(gray);
    let total: u64 = histogram.iter().map(|&count| u64::from(count)).sum();
    if total == 0 {
        return;
    }
    let value_at_rank = |rank: u64| {
        let mut seen = 0u64;
        for (value, &count) in histogram.iter().enumerate() {
            seen += u64::from(count);
            if seen > rank {
                return value as u8;
            }
        }
        255
    };
    let low = value_at_rank(total * 2 / 100);
    let high = value_at_rank((total * 98 / 100).min(total - 1));
    if high.saturating_sub(low) < 8 {
        return;
    }
    let (low, span) = (f32::from(low), f32::from(high - low));
    let lut: Vec<u8> = (0..=255u8)
        .map(|v| {
            ((f32::from(v) - low) / span * 255.0)
                .round()
                .clamp(0.0, 255.0) as u8
        })
        .collect();
    for pixel in gray.pixels_mut() {
        pixel[0] = lut[usize::from(pixel[0])];
    }
}

/// Enlarge a crop so its text is tall enough for OCR, preserving aspect
/// ratio. Uses Lanczos3, which keeps thin glyph strokes intact where a
/// nearest-neighbour blow-up would leave them jagged and unreadable.
/// Returns the image and the factor it was enlarged by (1 when unchanged).
fn upscale_for_ocr(image: GrayImage) -> (GrayImage, u32) {
    let height = image.height();
    if height == 0 || height >= MIN_OCR_TEXT_HEIGHT {
        return (image, 1);
    }
    // Integer factors avoid resampling artefacts on pixel-art UI text.
    let factor = MIN_OCR_TEXT_HEIGHT.div_ceil(height).clamp(2, 8);
    let (width, height) = (
        image.width().saturating_mul(factor),
        height.saturating_mul(factor),
    );
    (
        imageops::resize(&image, width, height, imageops::FilterType::Lanczos3),
        factor,
    )
}

/// Distinguishes the temp files of concurrent OCR calls within a process;
/// the process id distinguishes processes.
static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

/// Write `image` as a PNG to a new file in the temp directory.
///
/// Each call gets a name no other live call can hold — process id plus a
/// per-process counter — and the file is created exclusively, so two
/// threads or two processes OCR-ing at once can never read each other's
/// input. (A timestamp name could collide within one clock tick.)
fn write_temp_image(image: &DynamicImage) -> Option<TempImage> {
    write_temp_image_in(&ocr_temp_dir(), image)
}

/// Where the crops handed to Tesseract are written.
///
/// Tesseract opens its input with narrow-character file calls, which cannot
/// reach a path with non-ASCII characters — and on Windows the temporary
/// folder sits under the user's name, which need not be ASCII. There, the
/// crops go to the Public folder, which every Windows has under an ASCII
/// path, instead.
fn ocr_temp_dir() -> PathBuf {
    let temp = env::temp_dir();
    if cfg!(windows)
        && !temp.to_string_lossy().is_ascii()
        && let Some(public) = env::var_os("PUBLIC")
    {
        let dir = PathBuf::from(public).join("syrup").join("ocr");
        if fs::create_dir_all(&dir).is_ok() && dir.to_string_lossy().is_ascii() {
            return dir;
        }
    }
    temp
}

fn write_temp_image_in(dir: &Path, image: &DynamicImage) -> Option<TempImage> {
    let pid = std::process::id();
    for _ in 0..8 {
        let id = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
        let path = dir.join(format!("syrup-ocr-{pid}-{id}.png"));
        let file = match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => file,
            // Left behind by an earlier process that had this id; skip it.
            Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
            Err(_) => return None,
        };
        // Take ownership before writing: a failed write can still leave a
        // partial file on disk, and the guard cleans that up on `None`.
        let image_file = TempImage { path };
        let mut writer = BufWriter::new(file);
        image.write_to(&mut writer, ImageOutputFormat::Png).ok()?;
        // Flush, then close the file before Tesseract opens it.
        drop(writer.into_inner().ok()?);
        return Some(image_file);
    }
    None
}

fn find_tesseract_binary() -> Option<PathBuf> {
    if let Some(path) = env::var_os("TESSERACT_BIN") {
        let candidate = PathBuf::from(path);
        if candidate.is_file() {
            return Some(candidate);
        }
    }

    if let Some(path) = env::var_os("PATH") {
        for entry in env::split_paths(&path) {
            for name in ["tesseract.exe", "tesseract"] {
                let candidate = entry.join(name);
                if candidate.is_file() {
                    return Some(candidate);
                }
            }
        }
    }

    // The standard Windows installer locations, which are not on PATH by
    // default: machine-wide, and per-user when installed "for me only".
    // Anything else can be pointed at via TESSERACT_BIN.
    let mut candidates = vec![
        PathBuf::from(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        PathBuf::from(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    ];
    if let Some(local) = env::var_os("LOCALAPPDATA") {
        candidates.push(
            PathBuf::from(local)
                .join("Programs")
                .join("Tesseract-OCR")
                .join("tesseract.exe"),
        );
    }

    candidates.into_iter().find(|path| path.is_file())
}

fn normalize_text(text: &str) -> String {
    let mut normalized = text
        .replace('\r', "")
        .lines()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    normalized = normalized
        .chars()
        .map(|ch| match ch {
            '\u{2019}' | '\u{2018}' => '\'',
            '\u{2013}' | '\u{2014}' => '-',
            '\u{00A0}' => ' ',
            _ => ch,
        })
        .collect();
    normalized.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgba;

    fn as_strings(args: &[OsString]) -> Vec<String> {
        args.iter()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect()
    }

    #[test]
    fn tsv_config_comes_after_option_flags() {
        let args = as_strings(&tesseract_args(
            Path::new("crop.png"),
            &OcrConfig {
                psm: 11,
                whitelist: Some("0123456789".to_string()),
            },
        ));

        assert_eq!(args[0], "crop.png");
        assert_eq!(args[1], "stdout");
        // Tesseract stops parsing options at the first config-file argument,
        // so every flag must precede `tsv` or it is read as a config name.
        assert_eq!(args.last().map(String::as_str), Some("tsv"));
        let tsv = args.iter().position(|arg| arg == "tsv").unwrap();
        for flag in ["--oem", "--psm", "-c"] {
            let at = args.iter().position(|arg| arg == flag).unwrap();
            assert!(at < tsv, "{flag} must come before the tsv config file");
        }
        assert!(
            args.iter()
                .any(|arg| arg == "tessedit_char_whitelist=0123456789")
        );
    }

    #[test]
    fn config_helpers_reach_the_command_line() {
        let args = OcrConfig::single_line()
            .with_whitelist("0123456789/")
            .to_args();
        let psm = args.iter().position(|arg| arg == "--psm").unwrap();
        assert_eq!(args[psm + 1], "7");
        assert!(args.contains(&"tessedit_char_whitelist=0123456789/".to_string()));
    }

    #[test]
    fn temp_image_is_removed_on_drop() {
        let image = DynamicImage::ImageRgba8(RgbaImage::new(4, 4));
        let path = {
            let temp = write_temp_image(&image).expect("temp image written");
            let path = temp.path().to_path_buf();
            assert!(path.exists());
            path
        };
        assert!(!path.exists(), "temp OCR image outlived its guard");
    }

    /// Calls that overlap in time must never share an input file.
    #[test]
    fn concurrent_temp_images_get_distinct_files() {
        let threads: Vec<_> = (0..8)
            .map(|_| {
                std::thread::spawn(|| {
                    let image = DynamicImage::ImageRgba8(RgbaImage::new(2, 2));
                    (0..16)
                        .map(|_| write_temp_image(&image).expect("temp image written"))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        let held: Vec<TempImage> = threads
            .into_iter()
            .flat_map(|thread| thread.join().unwrap())
            .collect();
        let mut paths: Vec<&Path> = held.iter().map(TempImage::path).collect();
        paths.sort();
        paths.dedup();
        assert_eq!(paths.len(), 8 * 16);
    }

    #[test]
    fn a_leftover_file_is_never_overwritten() {
        let dir = env::temp_dir().join(format!("syrup-ocr-test-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let image = DynamicImage::ImageRgba8(RgbaImage::new(2, 2));
        // Occupy the name the next call would pick. Another test may take
        // that id first; the squatter must survive either way.
        let next = NEXT_TEMP_ID.load(Ordering::Relaxed);
        let squatter = dir.join(format!("syrup-ocr-{}-{next}.png", std::process::id()));
        fs::write(&squatter, b"not ours").unwrap();
        let temp = write_temp_image_in(&dir, &image).expect("temp image written");
        assert_ne!(temp.path(), squatter.as_path());
        assert_eq!(fs::read(&squatter).unwrap(), b"not ours");
        drop(temp);
        fs::remove_file(&squatter).unwrap();
        fs::remove_dir(&dir).unwrap();
    }

    #[test]
    fn crop_is_clipped_and_overflow_safe() {
        let mut image = RgbaImage::new(10, 8);
        image.put_pixel(9, 7, Rgba([1, 2, 3, 255]));
        let crop = crop_region(&image, 8, 6, u32::MAX, u32::MAX).expect("clipped crop");
        assert_eq!(crop.dimensions(), (2, 2));
        assert_eq!(crop.get_pixel(1, 1), &Rgba([1, 2, 3, 255]));
        assert!(crop_region(&image, u32::MAX, 0, 5, 5).is_none());
        assert!(crop_region(&image, 0, 0, 0, 5).is_none());
        assert!(crop_region(&image, 10, 0, 5, 5).is_none());
    }

    #[test]
    fn tsv_words_carry_boxes_and_confidence() {
        let tsv = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n\
                   1\t1\t0\t0\t0\t0\t0\t0\t300\t48\t-1\t\n\
                   4\t1\t1\t1\t1\t0\t12\t8\t200\t30\t-1\t\n\
                   5\t1\t1\t1\t1\t1\t12\t8\t90\t30\t96.5\tHP\n\
                   5\t1\t1\t1\t1\t2\t110\t8\t102\t30\t41.25\t1291/1351\n";
        let words = parse_tsv_words(tsv);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "HP");
        assert_eq!(
            (words[0].x, words[0].y, words[0].w, words[0].h),
            (12, 8, 90, 30)
        );
        assert!((words[0].confidence - 0.965).abs() < 1e-6);
        assert_eq!(words[1].text, "1291/1351");
        assert!((words[1].confidence - 0.4125).abs() < 1e-6);

        let result = OcrResult {
            text: "HP 1291/1351".into(),
            available: true,
            words,
        };
        assert!((result.confidence().value() - 0.68875).abs() < 1e-6);
        assert_eq!(OcrResult::default().confidence(), Confidence::NONE);
    }

    #[test]
    fn word_boxes_map_back_to_crop_pixels() {
        let word = OcrWord {
            text: "7".into(),
            x: 13,
            y: 0,
            w: 10,
            h: 47,
            confidence: 0.9,
        };
        // Enlarged 4x with no margin: pixels 13..23 cover crop columns 3..=5.
        let plain = Placement {
            scale: 4,
            margin: 0,
        };
        let mapped = unscale_word(word.clone(), plain, 20, 12);
        assert_eq!((mapped.x, mapped.y, mapped.w, mapped.h), (3, 0, 3, 12));

        // The same word read from a framed image sits `margin` further in.
        let framed = Placement {
            scale: 4,
            margin: 24,
        };
        let shifted = OcrWord {
            x: word.x + 24,
            y: word.y + 24,
            ..word
        };
        let mapped = unscale_word(shifted, framed, 20, 12);
        assert_eq!((mapped.x, mapped.y, mapped.w, mapped.h), (3, 0, 3, 12));
    }

    /// Light-on-dark text with a background at 30 and ink at 220.
    fn light_text_crop() -> RgbaImage {
        let mut crop = RgbaImage::from_pixel(40, 12, Rgba([30, 30, 34, 255]));
        for y in 3..9 {
            for x in 10..30 {
                crop.put_pixel(x, y, Rgba([220, 220, 225, 255]));
            }
        }
        crop
    }

    #[test]
    fn preprocessing_gives_dark_text_on_a_white_framed_page() {
        let (image, placement) = preprocess_image(&light_text_crop());
        let gray = image.to_luma8();
        // 12 rows need 4x to reach the OCR text height; the margin is half
        // the enlarged height.
        assert_eq!(
            placement,
            Placement {
                scale: 4,
                margin: 24
            }
        );
        assert_eq!(gray.dimensions(), (40 * 4 + 48, 12 * 4 + 48));
        let at = |x: u32, y: u32| {
            gray.get_pixel(placement.margin + x * 4 + 2, placement.margin + y * 4 + 2)[0]
        };
        assert_eq!(gray.get_pixel(0, 0)[0], 255, "margin is white");
        assert!(at(20, 6) < 30, "ink became dark: {}", at(20, 6));
        assert!(at(2, 1) > 225, "background became light: {}", at(2, 1));
    }

    #[test]
    fn dark_text_on_light_is_not_inverted() {
        let mut crop = light_text_crop();
        for pixel in crop.pixels_mut() {
            pixel.0 = [255 - pixel[0], 255 - pixel[1], 255 - pixel[2], 255];
        }
        let (image, placement) = preprocess_image(&crop);
        let gray = image.to_luma8();
        let at = |x: u32, y: u32| {
            gray.get_pixel(placement.margin + x * 4 + 2, placement.margin + y * 4 + 2)[0]
        };
        assert!(at(20, 6) < 30);
        assert!(at(2, 1) > 225);
    }

    #[test]
    fn faint_text_is_stretched_to_full_contrast() {
        let mut gray = GrayImage::from_pixel(50, 10, Luma([120]));
        for x in 10..40 {
            for y in 3..7 {
                gray.put_pixel(x, y, Luma([150]));
            }
        }
        stretch_levels(&mut gray);
        assert_eq!(gray.get_pixel(0, 0)[0], 0);
        assert_eq!(gray.get_pixel(20, 5)[0], 255);

        let mut flat = GrayImage::from_pixel(8, 8, Luma([90]));
        flat.put_pixel(3, 3, Luma([93]));
        stretch_levels(&mut flat);
        assert_eq!(flat.get_pixel(0, 0)[0], 90, "noise is not amplified");
    }

    #[test]
    fn tall_crops_are_framed_but_not_enlarged() {
        let crop = RgbaImage::from_pixel(30, 60, Rgba([200, 200, 200, 255]));
        let (image, placement) = preprocess_image(&crop);
        assert_eq!(
            placement,
            Placement {
                scale: 1,
                margin: 30
            }
        );
        assert_eq!((image.width(), image.height()), (90, 120));
    }

    /// Runs the real engine end to end: argument order, TSV parsing and the
    /// mapping of word boxes back onto the crop. Needs Tesseract installed,
    /// so it only runs on request: `cargo test -- --ignored`.
    #[test]
    #[ignore = "needs Tesseract installed"]
    fn reads_rendered_text_with_the_real_engine() {
        assert!(
            is_ocr_available(),
            "Tesseract not found: install it or set TESSERACT_BIN"
        );
        let (scale, x0, y0) = (2u32, 6u32, 4u32);
        let text = "SCORE 42";
        let width = crate::draw::text_width(text, scale) + 2 * x0;
        let height = crate::draw::text_height(scale) + 2 * y0;
        let mut image = RgbaImage::from_pixel(width + 30, height + 20, Rgba([24, 26, 30, 255]));
        crate::draw::draw_text(
            text,
            i64::from(x0 + 30),
            i64::from(y0 + 20),
            scale,
            |x, y| image.put_pixel(x as u32, y as u32, Rgba([235, 235, 235, 255])),
        );

        for config in [OcrConfig::default(), OcrConfig::single_line()] {
            let result = ocr_region_with(&image, 30, 20, width, height, &config)
                .expect("the engine reads clean text");
            assert_eq!(result.text, text, "psm {}", config.psm);
            assert!(result.confidence().value() > 0.5);
            // "SCORE" is drawn at x0..x0+58, y0..y0+14 within the crop.
            let score = &result.words[0];
            let close = |got: u32, want: u32| got.abs_diff(want) <= 2;
            assert!(
                close(score.x, x0) && close(score.y, y0),
                "box at {},{}",
                score.x,
                score.y
            );
            assert!(
                close(score.w, 58) && close(score.h, 14),
                "box {}x{}",
                score.w,
                score.h
            );
        }
    }

    /// A stand-in for Tesseract: a script that prints `stdout`, writes
    /// `stderr` and exits with `status`.
    #[cfg(unix)]
    fn fake_tesseract(name: &str, stdout: &str, stderr: &str, status: i32) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let dir = env::temp_dir().join(format!("syrup-fake-tesseract-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let tsv = dir.join(format!("{name}.tsv"));
        fs::write(&tsv, stdout).unwrap();
        let script = dir.join(name);
        fs::write(
            &script,
            format!(
                "#!/bin/sh\ncat '{}'\nprintf '%s' '{stderr}' >&2\nexit {status}\n",
                tsv.display()
            ),
        )
        .unwrap();
        fs::set_permissions(&script, fs::Permissions::from_mode(0o755)).unwrap();
        script
    }

    #[cfg(unix)]
    const TSV_HEADER: &str = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n";

    /// One test: the stand-ins are written and run in turn, never while
    /// another thread of this test binary might be starting one.
    #[cfg(unix)]
    #[test]
    fn recognize_maps_words_to_the_image_and_says_why_it_failed() {
        // A 60x12 crop is enlarged 4x and framed by a 24-pixel margin, so a
        // word Tesseract reports at (64, 32) 80x32 covers crop (10, 2) 20x8.
        let tsv = format!("{TSV_HEADER}5\t1\t1\t1\t1\t1\t64\t32\t80\t32\t88\tHP\n");
        let binary = fake_tesseract("reads", &tsv, "", 0);
        let image = RgbaImage::new(200, 100);
        let region = Rect {
            x: 10,
            y: 20,
            w: 60,
            h: 12,
        };
        let read = recognize_with(&binary, &image, region, &OcrConfig::default()).unwrap();
        assert_eq!(read.text, "HP");
        assert_eq!(
            read.words,
            vec![Word {
                text: "HP".into(),
                bounds: Rect {
                    x: 20,
                    y: 22,
                    w: 20,
                    h: 8
                },
                confidence: 0.88,
            }]
        );

        // No text is an empty recognition, not an error; a failure says why.
        let image = RgbaImage::new(40, 20);
        let all = Rect {
            x: 0,
            y: 0,
            w: 40,
            h: 20,
        };
        let silent = fake_tesseract("silent", TSV_HEADER, "", 0);
        assert_eq!(
            recognize_with(&silent, &image, all, &OcrConfig::default()),
            Ok(Recognition::default())
        );
        let broken = fake_tesseract("broken", "", "oops", 3);
        assert_eq!(
            recognize_with(&broken, &image, all, &OcrConfig::default()),
            Err(OcrError::EngineFailed {
                status: Some(3),
                stderr: "oops".into()
            })
        );
        let off_image = Rect {
            x: 50,
            y: 0,
            w: 10,
            h: 10,
        };
        assert_eq!(
            recognize_with(&silent, &image, off_image, &OcrConfig::default()),
            Err(OcrError::EmptyRegion)
        );
        assert_eq!(
            recognize_with(
                Path::new("/nonexistent/tesseract"),
                &image,
                all,
                &OcrConfig::default()
            )
            .map_err(|e| matches!(e, OcrError::EngineFailed { status: None, .. })),
            Err(true)
        );
    }
}
