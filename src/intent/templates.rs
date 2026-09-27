//! Pictures that `find_<name>_icon` intents look for.
//!
//! A template comes from the process (registered by name) or from a
//! `<name>.png` in the directory `$SYRUP_TEMPLATES` names. Registration
//! wins over the directory, so a program can override a file.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use image::RgbaImage;

static REGISTRY: RwLock<Option<HashMap<String, Arc<RgbaImage>>>> = RwLock::new(None);

/// Make `image` the picture behind `find_<name>_icon` (and `count_…`,
/// `track_…`) from now on. Intents resolved before this call keep the
/// picture they were resolved with.
pub fn register_template(name: &str, image: &RgbaImage) {
    let mut registry = REGISTRY.write().unwrap_or_else(|e| e.into_inner());
    registry
        .get_or_insert_with(HashMap::new)
        .insert(name.to_string(), Arc::new(image.clone()));
}

/// The picture registered under `name`, or `<name>.png` from
/// `$SYRUP_TEMPLATES`.
pub fn template_named(name: &str) -> Option<Arc<RgbaImage>> {
    if let Some(found) = REGISTRY
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .and_then(|r| r.get(name).cloned())
    {
        return Some(found);
    }
    let dir = PathBuf::from(std::env::var_os("SYRUP_TEMPLATES")?);
    let path = dir.join(format!("{name}.png"));
    image::open(path).ok().map(|i| Arc::new(i.to_rgba8()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registered_pictures_are_found_by_name_and_replaced_on_reregistration() {
        let one = RgbaImage::new(3, 3);
        register_template("unit-test-icon", &one);
        assert_eq!(
            template_named("unit-test-icon").map(|i| i.dimensions()),
            Some((3, 3))
        );
        let two = RgbaImage::new(5, 2);
        register_template("unit-test-icon", &two);
        assert_eq!(
            template_named("unit-test-icon").map(|i| i.dimensions()),
            Some((5, 2))
        );
        assert!(template_named("never-registered-icon").is_none());
    }
}
