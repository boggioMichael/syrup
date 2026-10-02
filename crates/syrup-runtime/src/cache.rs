//! Artifact store. Artifacts are built in a private staging directory and
//! published with a single rename, so readers only ever see complete ones.
//! Published artifacts are never modified; damaged ones are moved aside.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::abi::SYRUP_ABI_VERSION;
use crate::codegen::CODEGEN_VERSION;
use crate::compiler::{FLAGS, HOST_TARGET};
use crate::error::{ErrorKind, Result, Stage, SyrupError};
use crate::plan::hex;

const LAYOUT: &str = "v1";

/// Everything that decides whether a compiled module can be reused. The
/// operation name is deliberately absent, and so is the rustc version: the
/// module only talks C ABI, so any compiler's output stays valid.
pub fn artifact_key(plan_hash: &str) -> String {
    let identity = format!(
        "plan={plan_hash};codegen={CODEGEN_VERSION};abi={SYRUP_ABI_VERSION};target={HOST_TARGET};flags={}",
        FLAGS.join(" ")
    );
    hex(&Sha256::digest(identity))
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub key: String,
    pub intent: String,
    pub plan_hash: String,
    pub generator: String,
    pub abi_version: u32,
    pub target: String,
    pub rustc: String,
    pub source_sha256: String,
    pub library: String,
    pub library_sha256: String,
    pub built_at: u64,
    pub build_ms: u64,
    pub validation_cases: u32,
}

pub struct Store {
    root: PathBuf,
}

/// `bundle.json`: which operations a bundle holds and what it was built for.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BundleIndex {
    pub target: String,
    pub abi_version: u32,
    pub codegen: u32,
    /// Operation name to artifact key.
    pub operations: BTreeMap<String, String>,
}

impl BundleIndex {
    fn current() -> BundleIndex {
        BundleIndex {
            target: HOST_TARGET.to_string(),
            abi_version: SYRUP_ABI_VERSION,
            codegen: CODEGEN_VERSION,
            operations: BTreeMap::new(),
        }
    }

    /// Why this bundle cannot serve this build of Syrup, if it cannot.
    pub fn incompatibility(&self) -> Option<String> {
        if self.target != HOST_TARGET {
            Some(format!(
                "the bundle was built for {}, and this machine is {HOST_TARGET}",
                self.target
            ))
        } else if self.abi_version != SYRUP_ABI_VERSION || self.codegen != CODEGEN_VERSION {
            Some(format!(
                "the bundle was built by another Syrup version (ABI {}, codegen {}; this is ABI {SYRUP_ABI_VERSION}, codegen {CODEGEN_VERSION})",
                self.abi_version, self.codegen
            ))
        } else {
            None
        }
    }
}

pub fn sha256_file(path: &Path) -> io::Result<String> {
    Ok(hex(&Sha256::digest(fs::read(path)?)))
}

fn now() -> Duration {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
}

fn integrity(reason: String) -> SyrupError {
    SyrupError::new(Stage::Load, ErrorKind::Integrity, reason)
        .with_hint("the damaged artifact is moved aside and rebuilt in development mode; in frozen mode, prepare it again")
}

impl Store {
    pub fn new(root: PathBuf) -> Store {
        Store { root }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn artifact_dir(&self, key: &str) -> PathBuf {
        self.root.join(LAYOUT).join(key)
    }

    pub fn failed_dir(&self, key: &str) -> PathBuf {
        self.root.join("failed").join(key)
    }

    fn create(&self, dir: &Path) -> Result<()> {
        fs::create_dir_all(dir).map_err(|e| SyrupError::io(Stage::Compile, "cannot create", dir, e))
    }

    /// Exclusive across processes and threads; released when dropped.
    pub fn lock(&self, key: &str) -> Result<File> {
        let dir = self.root.join("locks");
        self.create(&dir)?;
        let path = dir.join(format!("{key}.lock"));
        let file = File::options()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&path)
            .map_err(|e| SyrupError::io(Stage::Compile, "cannot open", &path, e))?;
        file.lock()
            .map_err(|e| SyrupError::io(Stage::Compile, "cannot lock", &path, e))?;
        Ok(file)
    }

    pub fn staging(&self, key: &str) -> Result<PathBuf> {
        let dir = self.root.join("staging").join(format!(
            "{key}.{}.{}",
            std::process::id(),
            now().as_nanos()
        ));
        self.create(&dir)?;
        Ok(dir)
    }

    /// The published artifact for `key`, if any, after checking it is intact.
    pub fn open(&self, key: &str) -> Result<Option<(Manifest, PathBuf)>> {
        let dir = self.artifact_dir(key);
        let manifest_path = dir.join("manifest.json");
        let text = match fs::read_to_string(&manifest_path) {
            Ok(text) => text,
            Err(e) if e.kind() == io::ErrorKind::NotFound && !dir.exists() => return Ok(None),
            Err(e) => {
                return Err(integrity(format!(
                    "cannot read {}: {e}",
                    manifest_path.display()
                )));
            }
        };
        let manifest: Manifest = serde_json::from_str(&text).map_err(|e| {
            integrity(format!(
                "{} is not a valid manifest: {e}",
                manifest_path.display()
            ))
        })?;
        if manifest.key != key || manifest.abi_version != SYRUP_ABI_VERSION {
            return Err(integrity(format!(
                "{} does not describe artifact {key}",
                manifest_path.display()
            )));
        }
        let library = dir.join(&manifest.library);
        let actual = sha256_file(&library)
            .map_err(|e| integrity(format!("cannot read {}: {e}", library.display())))?;
        if actual != manifest.library_sha256 {
            return Err(integrity(format!(
                "{} does not match its manifest (sha256 {actual}, expected {})",
                library.display(),
                manifest.library_sha256
            ))
            .with_detail("library", library.display().to_string()));
        }
        Ok(Some((manifest, library)))
    }

    /// Returns the published directory. Losing a race to another builder is
    /// fine: both built the same thing, and the first one wins.
    pub fn publish(&self, staging: &Path, key: &str) -> Result<PathBuf> {
        let target = self.artifact_dir(key);
        self.create(target.parent().expect("artifact dirs have a parent"))?;
        let mut attempt = 0;
        loop {
            match fs::rename(staging, &target) {
                Ok(()) => return Ok(target),
                Err(_) if target.join("manifest.json").exists() => {
                    let _ = fs::remove_dir_all(staging);
                    return Ok(target);
                }
                // Windows can hold a just-unloaded library for a moment.
                Err(_) if attempt < 20 => {
                    attempt += 1;
                    thread::sleep(Duration::from_millis(25));
                }
                Err(e) => return Err(SyrupError::io(Stage::Compile, "cannot publish", &target, e)),
            }
        }
    }

    /// Keeps a failed build's source and logs for inspection.
    pub fn keep_failure(&self, staging: &Path, key: &str) -> PathBuf {
        let target = self.failed_dir(key);
        let _ = fs::remove_dir_all(&target);
        let _ = fs::create_dir_all(target.parent().expect("failed dirs have a parent"));
        if fs::rename(staging, &target).is_err() {
            let _ = fs::remove_dir_all(staging);
        }
        target
    }

    pub fn quarantine(&self, key: &str) -> Result<()> {
        let target = self
            .root
            .join("failed")
            .join(format!("{key}.damaged.{}", now().as_nanos()));
        self.create(target.parent().expect("failed dirs have a parent"))?;
        let dir = self.artifact_dir(key);
        fs::rename(&dir, &target)
            .map_err(|e| SyrupError::io(Stage::Load, "cannot move aside", &dir, e))
    }

    pub fn bundle_index(&self) -> Option<BundleIndex> {
        let text = fs::read_to_string(self.root.join("bundle.json")).ok()?;
        serde_json::from_str(&text).ok()
    }

    /// Copies published artifacts into the bundle at `dest`, adding to what
    /// is already there. `operations` maps names to keys published here.
    pub fn export(&self, dest: &Path, operations: &[(String, String)]) -> Result<BundleIndex> {
        let bundle = Store::new(dest.to_path_buf());
        let mut index = match bundle.bundle_index() {
            Some(index) => {
                if let Some(why) = index.incompatibility() {
                    return Err(SyrupError::new(
                        Stage::Compile,
                        ErrorKind::Conflicting,
                        format!("cannot add to {}: {why}", dest.display()),
                    ));
                }
                index
            }
            None => BundleIndex::current(),
        };
        for (name, key) in operations {
            let source = self.artifact_dir(key);
            let target = bundle.artifact_dir(key);
            if !target.join("manifest.json").exists() {
                let staging = bundle.staging(key)?;
                for file in fs::read_dir(&source)
                    .map_err(|e| SyrupError::io(Stage::Compile, "cannot read", &source, e))?
                {
                    let from = file
                        .map_err(|e| SyrupError::io(Stage::Compile, "cannot read", &source, e))?
                        .path();
                    let to = staging.join(from.file_name().expect("entries have names"));
                    fs::copy(&from, &to)
                        .map_err(|e| SyrupError::io(Stage::Compile, "cannot copy", &from, e))?;
                }
                bundle.publish(&staging, key)?;
            }
            bundle.open(key)?;
            index.operations.insert(name.clone(), key.clone());
        }
        let _ = fs::remove_dir_all(dest.join("staging"));
        let path = dest.join("bundle.json");
        fs::write(
            &path,
            serde_json::to_vec_pretty(&index).expect("indexes serialize"),
        )
        .map_err(|e| SyrupError::io(Stage::Compile, "cannot write", &path, e))?;
        Ok(index)
    }

    pub fn list(&self) -> Vec<Manifest> {
        let Ok(entries) = fs::read_dir(self.root.join(LAYOUT)) else {
            return vec![];
        };
        let mut manifests: Vec<Manifest> = entries
            .flatten()
            .filter_map(|entry| fs::read_to_string(entry.path().join("manifest.json")).ok())
            .filter_map(|text| serde_json::from_str(&text).ok())
            .collect();
        manifests.sort_by_key(|m| m.built_at);
        manifests
    }

    pub fn clear(&self) -> Result<()> {
        for dir in [LAYOUT, "staging", "failed", "locks"] {
            let path = self.root.join(dir);
            match fs::remove_dir_all(&path) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                Err(e) => return Err(SyrupError::io(Stage::Load, "cannot remove", &path, e)),
            }
        }
        Ok(())
    }
}
