use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;

use serde::Serialize;

macro_rules! named_enum {
    ($ty:ident { $($variant:ident => $name:literal,)* }) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
        #[serde(rename_all = "snake_case")]
        pub enum $ty {
            $($variant,)*
        }

        impl $ty {
            pub fn as_str(self) -> &'static str {
                match self {
                    $($ty::$variant => $name,)*
                }
            }
        }

        impl fmt::Display for $ty {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(self.as_str())
            }
        }
    };
}

named_enum!(Stage {
    Input => "input",
    Resolve => "resolve",
    Plan => "plan",
    Generate => "generate",
    Compile => "compile",
    Load => "load",
    Validate => "validate",
    Execute => "execute",
});

named_enum!(ErrorKind {
    BadImage => "bad_image",
    BadParameter => "bad_parameter",
    Malformed => "malformed",
    Ambiguous => "ambiguous",
    Unsupported => "unsupported",
    Conflicting => "conflicting",
    InvalidPlan => "invalid_plan",
    Policy => "policy",
    MissingDependency => "missing_dependency",
    PermissionDenied => "permission_denied",
    CompilerFailed => "compiler_failed",
    Timeout => "timeout",
    NotPrepared => "not_prepared",
    Integrity => "integrity",
    AbiMismatch => "abi_mismatch",
    Dlopen => "dlopen",
    Mismatch => "mismatch",
    ProviderFailed => "provider_failed",
    ModuleFailed => "module_failed",
    Panic => "panic",
    ContractViolation => "contract_violation",
    Io => "io",
});

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SyrupError {
    pub stage: Stage,
    pub kind: ErrorKind,
    pub operation: Option<String>,
    pub reason: String,
    pub hint: Option<String>,
    pub details: BTreeMap<String, String>,
}

impl SyrupError {
    pub fn new(stage: Stage, kind: ErrorKind, reason: impl Into<String>) -> Self {
        Self {
            stage,
            kind,
            operation: None,
            reason: reason.into(),
            hint: None,
            details: BTreeMap::new(),
        }
    }

    pub fn with_hint(mut self, hint: impl Into<String>) -> Self {
        self.hint = Some(hint.into());
        self
    }

    pub fn with_detail(mut self, key: &str, value: impl Into<String>) -> Self {
        self.details.insert(key.to_string(), value.into());
        self
    }

    pub fn for_operation(mut self, operation: &str) -> Self {
        self.operation.get_or_insert_with(|| operation.to_string());
        self
    }

    pub(crate) fn io(stage: Stage, what: &str, path: &Path, err: std::io::Error) -> Self {
        Self::new(
            stage,
            ErrorKind::Io,
            format!("{what} {}: {err}", path.display()),
        )
    }
}

impl fmt::Display for SyrupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}/{}] ", self.stage, self.kind)?;
        if let Some(operation) = &self.operation {
            write!(f, "{operation}: ")?;
        }
        f.write_str(&self.reason)?;
        if let Some(hint) = &self.hint {
            write!(f, " (hint: {hint})")?;
        }
        Ok(())
    }
}

impl std::error::Error for SyrupError {}

pub type Result<T> = std::result::Result<T, SyrupError>;
