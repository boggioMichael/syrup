//! Targets added at run time, each with its own detector, typically a model
//! called from Python. They get the same grammar, plans, generated code and
//! validation as built-in targets; only the detector call reaches them.

use std::sync::{Arc, RwLock};

use crate::catalog::{self, Capability, Finder, Name, Target, TargetEntry};
use crate::error::{ErrorKind, Result, Stage, SyrupError};
use crate::intent;
use crate::providers::Provider;

struct Added {
    entry: &'static TargetEntry,
    provider: Arc<dyn Provider>,
}

static ADDED: RwLock<Vec<Added>> = RwLock::new(Vec::new());

pub(crate) fn entries() -> Vec<&'static TargetEntry> {
    let added = ADDED.read().unwrap_or_else(|e| e.into_inner());
    added.iter().map(|a| a.entry).collect()
}

pub(crate) fn provider(name: Name) -> Option<Arc<dyn Provider>> {
    let added = ADDED.read().unwrap_or_else(|e| e.into_inner());
    added
        .iter()
        .find(|a| a.entry.target == Target::Custom(name))
        .map(|a| a.provider.clone())
}

fn leak(s: &str) -> &'static str {
    Box::leak(s.to_string().into_boxed_str())
}

/// Makes `singular`/`plural` a target found by `provider`, replacing an
/// earlier target with the same singular noun. The provider must return
/// scores in [0, 1].
pub fn add_target(
    singular: &str,
    plural: &str,
    default_min_confidence: f32,
    provider: Arc<dyn Provider>,
) -> Result<Target> {
    let error = |kind, reason: String| Err(SyrupError::new(Stage::Resolve, kind, reason));
    for noun in [singular, plural] {
        let well_formed = noun
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
            && noun.split('_').all(|w| !w.is_empty())
            && noun.bytes().next().is_some_and(|b| b.is_ascii_lowercase());
        if !well_formed {
            return error(
                ErrorKind::Malformed,
                format!("{noun:?} is not lowercase words joined by underscores"),
            );
        }
        if let Some(word) = noun.split('_').find(|w| intent::is_reserved(w)) {
            return error(
                ErrorKind::Conflicting,
                format!("{noun:?} uses {word:?}, which the grammar already gives a meaning"),
            );
        }
    }
    if !(0.0..=1.0).contains(&default_min_confidence) {
        return error(
            ErrorKind::Malformed,
            format!("min_confidence {default_min_confidence} is not in [0, 1]"),
        );
    }

    let mut added = ADDED.write().unwrap_or_else(|e| e.into_inner());
    let replacing = added.iter().position(|a| a.entry.singular[0] == singular);
    let others = catalog::TARGETS.iter().chain(
        added
            .iter()
            .enumerate()
            .filter(|(i, _)| Some(*i) != replacing)
            .map(|(_, a)| a.entry),
    );
    for other in others {
        let nouns = || other.singular.iter().chain(other.plural);
        if let Some(noun) = [singular, plural]
            .into_iter()
            .find(|n| nouns().any(|o| o == n))
        {
            return error(
                ErrorKind::Conflicting,
                format!("{noun:?} already names {}", other.plural[0]),
            );
        }
        if let Finder::Detect(c @ Capability::Custom(_)) = other.finder
            && c.abi_id() == catalog::custom_id(singular)
        {
            return error(
                ErrorKind::Conflicting,
                format!(
                    "{singular:?} collides with {:?}; choose another noun",
                    other.singular[0]
                ),
            );
        }
    }

    let name = Name(leak(singular));
    let entry: &'static TargetEntry = Box::leak(Box::new(TargetEntry {
        target: Target::Custom(name),
        label: name.as_str(),
        singular: Box::leak(Box::new([name.as_str()])),
        plural: Box::leak(Box::new([leak(plural)])),
        finder: Finder::Detect(Capability::Custom(name)),
        default_min_confidence,
        keypoints: &[],
        description: "added at run time",
    }));
    let new = Added { entry, provider };
    match replacing {
        Some(i) => added[i] = new,
        None => added.push(new),
    }
    Ok(Target::Custom(name))
}
