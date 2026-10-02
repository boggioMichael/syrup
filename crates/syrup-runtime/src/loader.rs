use std::ffi::CStr;
use std::path::{Path, PathBuf};

use libloading::Library;

use crate::abi::{SYRUP_ABI_VERSION, SyrupHost, SyrupImageView, SyrupParams, SyrupRunFn};
use crate::error::{ErrorKind, Result, Stage, SyrupError};

pub struct Module {
    run: SyrupRunFn,
    pub path: PathBuf,
    // Keeps `run` valid.
    _library: Library,
}

fn load_error(kind: ErrorKind, path: &Path, reason: String) -> SyrupError {
    SyrupError::new(Stage::Load, kind, reason).with_detail("library", path.display().to_string())
}

impl Module {
    /// Loads by absolute path only, never through the platform search path.
    pub fn load(path: &Path, plan_hash: &str) -> Result<Module> {
        let path = std::path::absolute(path)
            .map_err(|e| load_error(ErrorKind::Dlopen, path, format!("bad library path: {e}")))?;
        // SAFETY: generated modules have no initialisers; their source passed
        // the policy check before it was compiled.
        let library = unsafe { Library::new(&path) }.map_err(|e| {
            load_error(
                ErrorKind::Dlopen,
                &path,
                format!("cannot load {}: {e}", path.display()),
            )
        })?;
        let missing = |symbol: &str, e: libloading::Error| {
            load_error(
                ErrorKind::AbiMismatch,
                &path,
                format!("module does not export {symbol}: {e}"),
            )
        };
        // SAFETY: the signatures are those of ABI v1, checked right below.
        let (abi_version, hash, run) = unsafe {
            let abi_version = *library
                .get::<extern "C" fn() -> u32>(b"syrup_op_abi_version")
                .map_err(|e| missing("syrup_op_abi_version", e))?;
            let hash = *library
                .get::<extern "C" fn() -> *const std::ffi::c_char>(b"syrup_op_plan_hash")
                .map_err(|e| missing("syrup_op_plan_hash", e))?;
            let run = *library
                .get::<SyrupRunFn>(b"syrup_op_run")
                .map_err(|e| missing("syrup_op_run", e))?;
            (abi_version, hash, run)
        };
        let version = abi_version();
        if version != SYRUP_ABI_VERSION {
            return Err(load_error(
                ErrorKind::AbiMismatch,
                &path,
                format!("module speaks ABI {version}, the host speaks {SYRUP_ABI_VERSION}"),
            ));
        }
        // SAFETY: the module returns a pointer to a static NUL-terminated string.
        let found = unsafe { CStr::from_ptr(hash()) }.to_string_lossy();
        if found != plan_hash {
            return Err(load_error(
                ErrorKind::AbiMismatch,
                &path,
                format!("module implements plan {found}, expected {plan_hash}"),
            ));
        }
        Ok(Module {
            run,
            path,
            _library: library,
        })
    }

    /// # Safety
    /// `host.ctx` must be what `host`'s functions expect, and `input` must
    /// describe memory that stays readable for the call.
    pub unsafe fn run(
        &self,
        host: &SyrupHost,
        input: &SyrupImageView,
        params: &SyrupParams,
    ) -> i32 {
        // SAFETY: forwarded to the caller.
        unsafe { (self.run)(host, input, params) }
    }
}
