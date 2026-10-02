//! Compiles generated modules with `rustc` directly: no cargo, no build
//! scripts, no network.

use std::env;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use crate::error::{ErrorKind, Result, Stage, SyrupError};

pub const HOST_TARGET: &str = env!("SYRUP_HOST_TARGET");

// `#[unsafe(no_mangle)]` needs 1.82.
const MIN_MINOR: u32 = 82;

pub const FLAGS: &[&str] = &[
    "--edition=2021",
    "--crate-type=cdylib",
    "-Copt-level=3",
    "-Cdebuginfo=0",
    "-Cstrip=debuginfo",
    "-Cpanic=unwind",
    "-Dwarnings",
];

#[derive(Debug, Clone)]
pub struct Rustc {
    pub path: PathBuf,
    pub version: String,
}

fn missing(reason: String) -> SyrupError {
    SyrupError::new(Stage::Compile, ErrorKind::MissingDependency, reason).with_hint(
        "install Rust from https://rustup.rs or point SYRUP_RUSTC at a rustc binary; \
         deployments without a compiler can run prepared artifacts with SYRUP_MODE=frozen",
    )
}

impl Rustc {
    pub fn find(explicit: Option<&Path>) -> Result<Rustc> {
        let candidates: Vec<PathBuf> = match explicit {
            Some(path) => vec![path.to_path_buf()],
            None => {
                let mut list = vec![PathBuf::from("rustc")];
                let cargo_home = env::var_os("CARGO_HOME")
                    .map(PathBuf::from)
                    .or_else(|| env::home_dir().map(|home| home.join(".cargo")));
                if let Some(home) = cargo_home {
                    list.push(
                        home.join("bin")
                            .join(format!("rustc{}", env::consts::EXE_SUFFIX)),
                    );
                }
                list
            }
        };
        let mut tried = vec![];
        for path in &candidates {
            let output = match Command::new(path).arg("-vV").stdin(Stdio::null()).output() {
                Ok(output) if output.status.success() => output,
                Ok(output) => {
                    tried.push(format!("{}: exited with {}", path.display(), output.status));
                    continue;
                }
                Err(e) => {
                    tried.push(format!("{}: {e}", path.display()));
                    continue;
                }
            };
            let text = String::from_utf8_lossy(&output.stdout);
            let version = text.lines().next().unwrap_or_default().to_string();
            let minor = text
                .lines()
                .find_map(|line| line.strip_prefix("release: "))
                .and_then(|release| release.split('.').nth(1))
                .and_then(|minor| minor.parse::<u32>().ok());
            if minor.is_none_or(|minor| minor < MIN_MINOR) {
                tried.push(format!("{}: {version} is too old", path.display()));
                continue;
            }
            return Ok(Rustc {
                path: path.clone(),
                version,
            });
        }
        Err(
            missing(format!("no Rust compiler 1.{MIN_MINOR} or newer found"))
                .with_detail("tried", tried.join("; ")),
        )
    }

    /// Writes rustc's output to `rustc.log` next to the source.
    pub fn compile(
        &self,
        source: &Path,
        output: &Path,
        crate_name: &str,
        timeout: Duration,
    ) -> Result<()> {
        let dir = source
            .parent()
            .expect("sources live in a staging directory");
        let log_path = dir.join("rustc.log");
        let io = |e| SyrupError::io(Stage::Compile, "cannot write", &log_path, e);
        let log = File::create(&log_path).map_err(io)?;
        let mut child = Command::new(&self.path)
            .current_dir(dir)
            .args(FLAGS)
            .arg(format!("--target={HOST_TARGET}"))
            .arg(format!("--crate-name={crate_name}"))
            .arg(format!("--remap-path-prefix={}=syrup", dir.display()))
            .arg("-o")
            .arg(output)
            .arg(source)
            .stdin(Stdio::null())
            .stdout(log.try_clone().map_err(io)?)
            .stderr(log)
            .spawn()
            .map_err(|e| missing(format!("cannot run {}: {e}", self.path.display())))?;

        let started = Instant::now();
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break status,
                Ok(None) if started.elapsed() > timeout => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(SyrupError::new(
                        Stage::Compile,
                        ErrorKind::Timeout,
                        format!("rustc did not finish within {}s", timeout.as_secs()),
                    )
                    .with_hint("raise SYRUP_COMPILE_TIMEOUT_SECS on slow machines"));
                }
                Ok(None) => thread::sleep(Duration::from_millis(10)),
                Err(e) => return Err(missing(format!("lost track of rustc: {e}"))),
            }
        };
        if !status.success() {
            let log = fs::read_to_string(&log_path).unwrap_or_default();
            let start = (log.len().saturating_sub(4000)..log.len())
                .find(|&i| log.is_char_boundary(i))
                .unwrap_or(log.len());
            let tail = &log[start..];
            return Err(SyrupError::new(
                Stage::Compile,
                ErrorKind::CompilerFailed,
                format!("rustc rejected the generated module ({status})"),
            )
            .with_hint("this is a code generator bug; the source and compiler log are kept with the failed artifact")
            .with_detail("compiler_output", tail));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejected_source_reports_the_compiler_output() {
        let dir = env::temp_dir().join(format!("syrup-compiler-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("op.rs"), "fn broken( {").unwrap();
        let rustc = Rustc::find(None).unwrap();
        let e = rustc
            .compile(
                &dir.join("op.rs"),
                &dir.join("out"),
                "op",
                Duration::from_secs(300),
            )
            .unwrap_err();
        let _ = fs::remove_dir_all(&dir);
        assert_eq!(
            (e.stage, e.kind),
            (Stage::Compile, ErrorKind::CompilerFailed)
        );
        assert!(e.details["compiler_output"].contains("error"), "{e:?}");
    }
}
