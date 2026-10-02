//! Turns operation names such as `find_faces_in_top_half` into native code:
//! the name is parsed into an intent, the intent into a typed plan, the plan
//! into a small Rust `cdylib` that is compiled with `rustc`, checked against
//! a reference interpreter, cached, and loaded.
//!
//! ```no_run
//! use syrup_runtime::{ImageInput, RunParams, Runtime};
//!
//! let runtime = Runtime::from_env()?;
//! let find_face = runtime.resolve("find_face")?;
//! let image = image::open("photo.jpg").unwrap().to_rgb8();
//! let faces = find_face.run(&ImageInput::from_rgb(&image), &RunParams::default())?;
//! for face in &faces.items {
//!     println!("{:?} {:.2}", face.bbox, face.confidence);
//! }
//! # Ok::<(), syrup_runtime::SyrupError>(())
//! ```

pub mod abi {
    include!("abi_prelude.rs");
}

mod cache;
pub mod catalog;
mod codegen;
mod compiler;
pub mod contract;
pub mod custom;
pub mod error;
pub mod frames;
mod host;
pub mod intent;
pub mod interp;
mod loader;
pub mod plan;
pub mod providers;
mod runtime;
mod validate;

pub use cache::{BundleIndex, Manifest, Store};
pub use contract::{
    ArtifactStatus, BoxF, FindResult, Found, ImageInput, Keypoint, OwnedImage, RunParams, TrackRef,
};
pub use error::{ErrorKind, Result, Stage, SyrupError};
pub use intent::{Intent, NormRect, OrderKey, PixelRect, Ratio, RegionSpec};
pub use runtime::{Config, Mode, Operation, Prepared, Runtime, Session, SessionOptions};
