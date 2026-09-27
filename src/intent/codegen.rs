//! Writing an intent's implementation out as a Rust crate.
//!
//! The generated crate is deliberately small and readable: one `run`
//! function that calls the same [`plans`](super::plans) functions the
//! in-process executor uses, with every parameter spelled out, and one
//! [`export_intent!`](crate::export_intent) line for the C ABI. It is what
//! the library "invented" for the name, in a form a developer can read,
//! copy into their own code, or edit and rebuild.
//!
//! A plan that needs a picture gets it embedded (`include_bytes!` of a PNG
//! written next to the source), and a plan that keeps state between calls
//! gets a `static` for it, so the built library is self-contained.

use std::path::Path;

use super::{Half, Pick, Plan};

/// A generated crate: its `src/lib.rs`, and any files to write next to it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Generated {
    pub source: String,
    /// `(file name under src/, bytes)`.
    pub extra_files: Vec<(String, Vec<u8>)>,
}

/// What `name` does, in a sentence.
pub fn describe(plan: &Plan) -> String {
    match plan {
        Plan::Faces { profile: false, .. } => "runs the bundled frontal-face cascade over the region and reports every face, best-supported first".into(),
        Plan::Faces { profile: true, .. } => "runs the bundled profile-face cascade over the region and reports every face turned to the side".into(),
        Plan::Eyes { .. } => "finds frontal faces, then runs the eye cascade inside each".into(),
        Plan::ColorBars { color } => format!("finds the {} regions of the region that are much wider than tall", color.name()),
        Plan::ColorBlobs { color } => format!("finds every {} region, largest first", color.name()),
        Plan::TextBlock => "finds the block of text-like pixels".into(),
        Plan::Icon { name, .. } => format!("finds every place the picture `{name}` appears, by normalised correlation"),
        Plan::Motion => "reports what moved since the previous call, with stable ids".into(),
        Plan::BarFill { color } => format!("measures how full the largest {} bar is, as a percentage of its track", color.name()),
        Plan::Ocr => "reads the region's text through the OCR engine".into(),
        Plan::Within { half, inner } => format!("{}, in the {} half of the region", describe(inner), half_name(*half)),
        Plan::Keep { pick, inner } => format!("{}, keeping only the {}", describe(inner), pick_name(*pick)),
        Plan::Count { inner } => format!("counts how many the plan finds that {}", describe(inner)),
        Plan::Track { inner } => format!("{}, with a stable id per object across calls", describe(inner)),
    }
}

fn half_name(half: Half) -> &'static str {
    match half {
        Half::Top => "top",
        Half::Bottom => "bottom",
        Half::Left => "left",
        Half::Right => "right",
    }
}

fn pick_name(pick: Pick) -> &'static str {
    match pick {
        Pick::Largest => "largest",
        Pick::Smallest => "smallest",
    }
}

/// The expression computing a plan's result, given `image`, `region` and
/// (for stateful plans) `state` in scope. Statements go in `lines`.
fn expression(plan: &Plan, lines: &mut Vec<String>, files: &mut Vec<(String, Vec<u8>)>) -> String {
    match plan {
        Plan::Faces { profile, options } => format!(
            "plans::find_faces(\n        image,\n        region,\n        {profile},\n        &CascadeOptions {{\n            scale_factor: {:?},\n            min_neighbors: {},\n            min_size: {},\n            max_size: {:?},\n            ..CascadeOptions::default()\n        }},\n    )",
            options.scale_factor, options.min_neighbors, options.min_size, options.max_size
        ),
        Plan::Eyes { options } => format!(
            "plans::find_eyes(\n        image,\n        region,\n        &CascadeOptions {{\n            scale_factor: {:?},\n            min_neighbors: {},\n            min_size: {},\n            max_size: {:?},\n            ..CascadeOptions::default()\n        }},\n    )",
            options.scale_factor, options.min_neighbors, options.min_size, options.max_size
        ),
        Plan::ColorBars { color } => format!("plans::find_bars(image, region, {:?})", color.hue()),
        Plan::ColorBlobs { color } => {
            format!("plans::find_blobs(image, region, {:?})", color.hue())
        }
        Plan::TextBlock => "plans::find_text(image, region)".into(),
        Plan::Icon { name, template } => {
            let file = format!("{}.png", safe_file_name(name));
            let mut png = Vec::new();
            let encoder = image::codecs::png::PngEncoder::new(&mut png);
            image::ImageEncoder::write_image(
                encoder,
                template.as_raw(),
                template.width(),
                template.height(),
                image::ColorType::Rgba8,
            )
            .expect("encoding an in-memory image to PNG cannot fail");
            files.push((file.clone(), png));
            lines.push(format!(
                "    static PICTURE: std::sync::OnceLock<RgbaImage> = std::sync::OnceLock::new();\n    let picture = PICTURE.get_or_init(|| {{\n        syrup::image::load_from_memory(include_bytes!(\"{file}\"))\n            .expect(\"the embedded picture decodes\")\n            .to_rgba8()\n    }});"
            ));
            "plans::find_icon(image, region, picture)".into()
        }
        Plan::Motion => "plans::find_motion(&mut state, image, region)".into(),
        Plan::BarFill { color } => format!("plans::measure_bar(image, region, {:?})", color.hue()),
        Plan::Ocr => "plans::read_text(image, region)".into(),
        Plan::Within { half, inner } => {
            lines.push(format!("    let region = Half::{:?}.of(region);", half));
            expression(inner, lines, files)
        }
        Plan::Keep { pick, inner } => {
            let inner = expression(inner, lines, files);
            format!("plans::keep(Pick::{pick:?}, {inner})")
        }
        Plan::Count { inner } => {
            let inner = expression(inner, lines, files);
            format!("plans::count({inner})")
        }
        Plan::Track { inner } => {
            let inner = expression(inner, lines, files);
            lines.push(format!("    let found = {inner};"));
            "plans::track(&mut state, found)".into()
        }
    }
}

fn safe_file_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// The crate implementing `name` with `plan`.
pub fn generate(name: &str, plan: &Plan) -> Generated {
    let mut lines = Vec::new();
    let mut files = Vec::new();
    let result = expression(plan, &mut lines, &mut files);
    let description = describe(plan);

    let mut imports = String::new();
    if matches!(plan_leaf(plan), Plan::Faces { .. } | Plan::Eyes { .. }) {
        imports.push_str("use syrup::cascade::CascadeOptions;\n");
    }
    if contains(plan, |p| matches!(p, Plan::Within { .. })) {
        imports.push_str("use syrup::intent::Half;\n");
    }
    if contains(plan, |p| matches!(p, Plan::Keep { .. })) {
        imports.push_str("use syrup::intent::Pick;\n");
    }
    if plan.is_stateful() {
        imports.push_str("use syrup::intent::State;\n");
        lines.insert(
            0,
            "    static STATE: std::sync::Mutex<State> = std::sync::Mutex::new(State::new());\n    let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());".into(),
        );
    }
    let body = if lines.is_empty() {
        format!("    {result}")
    } else {
        format!("{}\n    {result}", lines.join("\n"))
    };
    let source = format!(
        "//! `{name}`, as syrup {version} understood it: {description}.\n\
         //!\n\
         //! Generated by `syrup::intent`. Edit it if you like — it is ordinary Rust\n\
         //! over the library's primitives — or delete the directory to regenerate.\n\
         \n\
         {imports}\
         use syrup::image::RgbaImage;\n\
         use syrup::intent::{{Outcome, plans}};\n\
         use syrup::{{Detection, Rect}};\n\
         \n\
         /// {description_cap}.\n\
         pub fn run(image: &RgbaImage, region: Rect) -> Detection<Outcome> {{\n\
         {body}\n\
         }}\n\
         \n\
         syrup::export_intent!(\"{name}\", run);\n",
        version = env!("CARGO_PKG_VERSION"),
        description_cap = capitalise(&description),
    );
    Generated {
        source,
        extra_files: files,
    }
}

/// The `src/lib.rs` of the crate implementing `name` with `plan`.
pub fn source(name: &str, plan: &Plan) -> String {
    generate(name, plan).source
}

fn plan_leaf(plan: &Plan) -> &Plan {
    match plan {
        Plan::Within { inner, .. }
        | Plan::Keep { inner, .. }
        | Plan::Count { inner }
        | Plan::Track { inner } => plan_leaf(inner),
        leaf => leaf,
    }
}

fn contains(plan: &Plan, predicate: fn(&Plan) -> bool) -> bool {
    predicate(plan)
        || match plan {
            Plan::Within { inner, .. }
            | Plan::Keep { inner, .. }
            | Plan::Count { inner }
            | Plan::Track { inner } => contains(inner, predicate),
            _ => false,
        }
}

/// The `Cargo.toml` of the crate implementing `name`, depending on the
/// syrup sources at `syrup_dir`.
pub fn manifest(name: &str, syrup_dir: &Path) -> String {
    // Forward slashes keep the manifest valid on every platform, and a
    // literal string avoids escaping the rest.
    let path = syrup_dir
        .to_string_lossy()
        .replace('\\', "/")
        .replace('\'', "");
    format!(
        "[package]\n\
         name = \"{crate}\"\n\
         version = \"0.1.0\"\n\
         edition = \"2024\"\n\
         publish = false\n\
         \n\
         [lib]\n\
         crate-type = [\"cdylib\"]\n\
         \n\
         [dependencies]\n\
         syrup = {{ path = '{path}' }}\n\
         \n\
         [profile.release]\n\
         codegen-units = 16\n\
         debug = false\n",
        crate = crate_name(name),
    )
}

/// The package name of the crate implementing `name`.
pub fn crate_name(name: &str) -> String {
    format!("syrup_intent_{name}")
}

fn capitalise(text: &str) -> String {
    let mut chars = text.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cascade::CascadeOptions;
    use crate::intent::Color;

    #[test]
    fn generated_source_names_the_plan_and_exports_the_abi() {
        let text = source(
            "find_face",
            &Plan::Faces {
                profile: false,
                options: CascadeOptions::default(),
            },
        );
        assert!(text.contains("plans::find_faces("));
        assert!(text.contains("min_neighbors: 3"));
        assert!(text.contains("syrup::export_intent!(\"find_face\", run);"));
        assert!(text.starts_with("//! `find_face`, as syrup"));

        let text = source("measure_red_bar", &Plan::BarFill { color: Color::Red });
        assert!(text.contains("plans::measure_bar(image, region, (340.0, 20.0))"));
    }

    #[test]
    fn wrapped_plans_nest_and_stateful_plans_get_a_static() {
        let plan = Plan::Within {
            half: Half::Bottom,
            inner: Box::new(Plan::Count {
                inner: Box::new(Plan::Keep {
                    pick: Pick::Largest,
                    inner: Box::new(Plan::ColorBlobs {
                        color: Color::Green,
                    }),
                }),
            }),
        };
        let text = source("count_largest_green_blob_bottom", &plan);
        assert!(text.contains("let region = Half::Bottom.of(region);"));
        assert!(text.contains("plans::count(plans::keep(Pick::Largest, plans::find_blobs(image, region, (70.0, 170.0))))"));
        assert!(text.contains("use syrup::intent::Half;"));
        assert!(!text.contains("State"));

        let plan = Plan::Track {
            inner: Box::new(Plan::ColorBlobs { color: Color::Red }),
        };
        let text = source("track_red_blob", &plan);
        assert!(text.contains("static STATE: std::sync::Mutex<State>"));
        assert!(text.contains("let found = plans::find_blobs(image, region, (340.0, 20.0));"));
        assert!(text.contains("plans::track(&mut state, found)"));
    }

    #[test]
    fn icons_embed_their_picture() {
        let picture = image::RgbaImage::from_fn(4, 4, |x, y| {
            image::Rgba([x as u8 * 60, y as u8 * 60, 0, 255])
        });
        let plan = Plan::Icon {
            name: "boss".into(),
            template: std::sync::Arc::new(picture.clone()),
        };
        let generated = generate("find_boss_icon", &plan);
        assert!(generated.source.contains("include_bytes!(\"boss.png\")"));
        assert_eq!(generated.extra_files.len(), 1);
        assert_eq!(generated.extra_files[0].0, "boss.png");
        let decoded = image::load_from_memory(&generated.extra_files[0].1)
            .unwrap()
            .to_rgba8();
        assert_eq!(decoded, picture);
    }

    #[test]
    fn manifests_point_back_at_the_library() {
        let text = manifest("find_face", Path::new("C:\\Users\\dev\\syrup"));
        assert!(text.contains("name = \"syrup_intent_find_face\""));
        assert!(text.contains("syrup = { path = 'C:/Users/dev/syrup' }"));
        assert!(text.contains("crate-type = [\"cdylib\"]"));
    }
}
