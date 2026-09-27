//! Compiling an intent's crate to a shared library and loading it.
//!
//! ```text
//!   source ─▶ <cache>/<name>-<hash>/{Cargo.toml, src/lib.rs}
//!          ─▶ cargo build --release  (target dir shared across intents)
//!          ─▶ <target>/release/libsyrup_intent_<name>.so | .dll | .dylib
//!          ─▶ libloading, ABI version check, name check
//! ```
//!
//! The cache lives in `$SYRUP_INTENT_CACHE`, else the user's cache
//! directory, else the system temp directory. A rebuild only happens when
//! the generated source or the library version changes, because the
//! directory name carries a hash of both.

use std::ffi::{CStr, c_char};
use std::path::{Path, PathBuf};
use std::process::Command;

use image::RgbaImage;

use crate::abi::{self, FrameView, RegionView, ResultView};
use crate::detection::Detection;
use crate::geometry::Rect;
use crate::intent::{IntentError, Outcome, codegen};

type VersionFn = unsafe extern "C" fn() -> u32;
type NameFn = unsafe extern "C" fn() -> *const c_char;
type RunFn = unsafe extern "C" fn(*const FrameView, *const RegionView, *mut ResultView) -> i32;
type FreeFn = unsafe extern "C" fn(*mut ResultView);

/// A loaded implementation of one intent.
pub struct Library {
    path: PathBuf,
    run: RunFn,
    free: FreeFn,
    // Dropped last: the function pointers above point into it.
    _library: libloading::Library,
}

impl std::fmt::Debug for Library {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Library").field("path", &self.path).finish()
    }
}

impl Library {
    /// Load a shared library that exports the intent ABI for `name`.
    pub fn load(name: &str, path: &Path) -> Result<Self, IntentError> {
        let failed = |reason: String| IntentError::LoadFailed {
            name: name.to_string(),
            reason,
        };
        // SAFETY: loading runs the library's initialisers; a library built
        // by this module from generated source has none beyond Rust's.
        let library = unsafe { libloading::Library::new(path) }
            .map_err(|e| failed(format!("{}: {e}", path.display())))?;
        // SAFETY: the symbol types match the exports of `export_intent!`.
        let (version, exported_name, run, free) = unsafe {
            let version: libloading::Symbol<VersionFn> = library
                .get(abi::SYMBOL_VERSION)
                .map_err(|e| failed(format!("no syrup_abi_version export: {e}")))?;
            let exported: libloading::Symbol<NameFn> = library
                .get(abi::SYMBOL_NAME)
                .map_err(|e| failed(format!("no syrup_intent_name export: {e}")))?;
            let run: libloading::Symbol<RunFn> = library
                .get(abi::SYMBOL_RUN)
                .map_err(|e| failed(format!("no syrup_intent_run export: {e}")))?;
            let free: libloading::Symbol<FreeFn> = library
                .get(abi::SYMBOL_FREE)
                .map_err(|e| failed(format!("no syrup_intent_free export: {e}")))?;
            let exported_name = CStr::from_ptr(exported()).to_string_lossy().into_owned();
            (version(), exported_name, *run, *free)
        };
        if version != abi::ABI_VERSION {
            return Err(failed(format!(
                "built against ABI version {version}, this library speaks {}",
                abi::ABI_VERSION
            )));
        }
        if exported_name != name {
            return Err(failed(format!(
                "the library implements `{exported_name}`, not `{name}`"
            )));
        }
        Ok(Self {
            path: path.to_path_buf(),
            run,
            free,
            _library: library,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Run the implementation on `region` of `image`.
    pub fn run(&self, image: &RgbaImage, region: Rect) -> Detection<Outcome> {
        let frame = FrameView::of(image);
        let region = RegionView::from(region);
        let mut out = ResultView::empty();
        // SAFETY: the views are valid for the duration of the call and the
        // library follows the ABI contract (checked at load).
        let code = unsafe { (self.run)(&frame, &region, &mut out) };
        if code != 0 {
            return Detection::missing(
                "intent",
                format!(
                    "the native implementation failed (code {code}: {})",
                    match code {
                        1 => "bad arguments",
                        2 => "it panicked",
                        _ => "unknown",
                    }
                ),
            );
        }
        // SAFETY: `out` was filled by the library's run and is released by
        // the same library's free, exactly once.
        let detection = unsafe { out.to_detection("intent") };
        unsafe { (self.free)(&mut out) };
        detection
    }
}

/// Where compiled intents are kept.
pub fn cache_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("SYRUP_INTENT_CACHE") {
        return PathBuf::from(dir);
    }
    let base = if cfg!(windows) {
        std::env::var_os("LOCALAPPDATA").map(PathBuf::from)
    } else {
        std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
    };
    base.unwrap_or_else(std::env::temp_dir)
        .join("syrup")
        .join("intents")
}

/// Where the library's own sources are, for the generated crate to depend
/// on: `$SYRUP_SOURCE_DIR`, else the directory this build came from.
pub fn syrup_source_dir() -> PathBuf {
    std::env::var_os("SYRUP_SOURCE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")))
}

/// FNV-1a over the inputs that decide what gets built.
fn fingerprint(parts: &[&str]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for part in parts {
        for byte in part.bytes().chain(std::iter::once(0)) {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
    }
    hash
}

/// The path the built library will have, for `name` in `crate_dir`.
fn artifact_path(target_dir: &Path, name: &str) -> PathBuf {
    target_dir.join("release").join(format!(
        "{}{}{}",
        std::env::consts::DLL_PREFIX,
        codegen::crate_name(name),
        std::env::consts::DLL_SUFFIX
    ))
}

fn fingerprint_bytes(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

fn write_if_changed(path: &Path, content: &str) -> std::io::Result<()> {
    if std::fs::read_to_string(path).ok().as_deref() == Some(content) {
        return Ok(());
    }
    std::fs::write(path, content)
}

/// Write the crate for `name` into the cache, build it, and load it.
pub fn build_and_load(name: &str, generated: &codegen::Generated) -> Result<Library, IntentError> {
    let syrup_dir = syrup_source_dir();
    let manifest = codegen::manifest(name, &syrup_dir);
    let source = &generated.source;
    let extra_stamp: String = generated
        .extra_files
        .iter()
        .map(|(file, bytes)| format!("{file}:{:016x}", fingerprint_bytes(bytes)))
        .collect::<Vec<_>>()
        .join(",");
    let stamp = fingerprint(&[
        source,
        &manifest,
        &extra_stamp,
        abi::ABI_VERSION.to_string().as_str(),
    ]);
    let cache = cache_dir();
    let crate_dir = cache.join(format!("{name}-{stamp:016x}"));
    let target_dir = cache.join("target");
    let failed = |log: String| IntentError::BuildFailed {
        name: name.to_string(),
        log,
    };

    std::fs::create_dir_all(crate_dir.join("src"))
        .map_err(|e| failed(format!("cannot create {}: {e}", crate_dir.display())))?;
    write_if_changed(&crate_dir.join("Cargo.toml"), &manifest)
        .map_err(|e| failed(e.to_string()))?;
    write_if_changed(&crate_dir.join("src").join("lib.rs"), source)
        .map_err(|e| failed(e.to_string()))?;
    for (file, bytes) in &generated.extra_files {
        let path = crate_dir.join("src").join(file);
        if std::fs::read(&path).ok().as_deref() != Some(bytes.as_slice()) {
            std::fs::write(&path, bytes)
                .map_err(|e| failed(format!("cannot write {}: {e}", path.display())))?;
        }
    }
    // Pin the same dependency versions as the library itself, so an
    // offline machine (or a sandbox) resolves without the network.
    if let Ok(lock) = std::fs::read_to_string(syrup_dir.join("Cargo.lock")) {
        let _ = write_if_changed(&crate_dir.join("Cargo.lock"), &lock);
    }

    let artifact = artifact_path(&target_dir, name);
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let mut command = Command::new(cargo);
    command
        .arg("build")
        .arg("--release")
        .arg("--quiet")
        .current_dir(&crate_dir)
        .env("CARGO_TARGET_DIR", &target_dir)
        // Cargo variables inherited from a `cargo test` or `cargo run` parent
        // must not leak into the nested build.
        .env_remove("CARGO_MANIFEST_DIR")
        .env_remove("CARGO_PKG_NAME");
    if std::env::var_os("SYRUP_OFFLINE").is_some()
        || std::env::var_os("CARGO_NET_OFFLINE").is_some_and(|v| v == "true")
    {
        command.arg("--offline");
    }
    let output = command.output().map_err(|e| {
        failed(format!(
            "cannot run cargo: {e}; a native intent needs a Rust toolchain on this machine"
        ))
    })?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let tail: String = stderr
            .chars()
            .rev()
            .take(4000)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        return Err(failed(format!(
            "cargo build in {} failed:\n{tail}",
            crate_dir.display()
        )));
    }
    if !artifact.is_file() {
        return Err(failed(format!(
            "cargo reported success but {} does not exist",
            artifact.display()
        )));
    }
    Library::load(name, &artifact)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprints_change_with_content() {
        assert_ne!(fingerprint(&["a"]), fingerprint(&["b"]));
        assert_ne!(fingerprint(&["a", "b"]), fingerprint(&["ab"]));
        assert_eq!(fingerprint(&["x"]), fingerprint(&["x"]));
    }

    #[test]
    fn artifact_paths_follow_the_platform() {
        let path = artifact_path(Path::new("/t"), "find_face");
        let file = path.file_name().unwrap().to_string_lossy();
        assert!(file.contains("syrup_intent_find_face"));
        assert!(file.ends_with(std::env::consts::DLL_SUFFIX));
    }

    #[test]
    fn loading_something_that_is_not_a_library_fails_clearly() {
        let dir = std::env::temp_dir().join("syrup-not-a-library");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("nope.txt");
        std::fs::write(&path, "hello").unwrap();
        match Library::load("find_face", &path) {
            Err(IntentError::LoadFailed { name, reason }) => {
                assert_eq!(name, "find_face");
                assert!(reason.contains("nope.txt"), "{reason}");
            }
            other => panic!("expected LoadFailed, got {other:?}"),
        }
    }
}
