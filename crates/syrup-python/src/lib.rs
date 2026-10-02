//! The one native entry point the Python package uses. Operations are not
//! bound one by one: Python passes a name and pixels, and gets JSON back.

use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::PyBytes;
use serde::Deserialize;
use syrup_runtime::abi::SyrupDetection;
use syrup_runtime::contract::ProviderInfo;
use syrup_runtime::frames::{FrameSource, WindowCapture};
use syrup_runtime::intent::{region_named, region_names};
use syrup_runtime::providers::{Detections, Provider, ViewRef};
use syrup_runtime::{
    ErrorKind, ImageInput, Intent, NormRect, Operation, OrderKey, OwnedImage, PixelRect, Ratio,
    RegionSpec, RunParams, Runtime, Session, SessionOptions, Stage, SyrupError, catalog,
};

create_exception!(_native, NativeError, PyException);

fn raise(e: SyrupError) -> PyErr {
    NativeError::new_err(serde_json::to_string(&e).expect("errors serialize"))
}

fn runtime() -> PyResult<&'static Runtime> {
    static RUNTIME: OnceLock<Result<Runtime, SyrupError>> = OnceLock::new();
    RUNTIME
        .get_or_init(Runtime::from_env)
        .as_ref()
        .map_err(|e| raise(e.clone()))
}

#[pyclass(frozen, module = "syrup._native")]
struct NativeOperation(Operation);

#[pymethods]
impl NativeOperation {
    #[getter]
    fn name(&self) -> &str {
        self.0.name()
    }

    #[getter]
    fn intent(&self) -> String {
        self.0.intent().to_string()
    }

    #[getter]
    fn plan_hash(&self) -> &str {
        self.0.plan_hash()
    }

    #[getter]
    fn artifact_key(&self) -> &str {
        self.0.artifact_key()
    }

    #[getter]
    fn needs_region(&self) -> bool {
        self.0.plan().needs_caller_region()
    }

    fn plan(&self) -> String {
        serde_json::to_string(self.0.plan()).expect("plans serialize")
    }

    fn source(&self) -> String {
        self.0.source()
    }

    fn explain(&self) -> String {
        self.0.explain()
    }

    fn prepare(&self, py: Python<'_>) -> PyResult<String> {
        let prepared = py.detach(|| self.0.prepare()).map_err(raise)?;
        Ok(serde_json::json!({
            "status": prepared.status,
            "library": prepared.library,
            "elapsed_ms": prepared.elapsed.as_secs_f64() * 1000.0,
            "manifest": prepared.manifest,
        })
        .to_string())
    }

    #[pyo3(signature = (max_distance=None, grace_frames=None))]
    fn session(
        &self,
        max_distance: Option<f32>,
        grace_frames: Option<u32>,
    ) -> PyResult<NativeSession> {
        let defaults = SessionOptions::default();
        let options = SessionOptions {
            max_distance: max_distance.unwrap_or(defaults.max_distance),
            grace_frames: grace_frames.unwrap_or(defaults.grace_frames),
        };
        let session = self.0.session(options).map_err(raise)?;
        Ok(NativeSession(Mutex::new(session)))
    }

    #[pyo3(signature = (pixels, width, height, channels, min_confidence=None, max_results=None, region=None))]
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        py: Python<'_>,
        pixels: PyBackedBytes,
        width: u32,
        height: u32,
        channels: u32,
        min_confidence: Option<f32>,
        max_results: Option<u32>,
        region: Option<(u32, u32, u32, u32)>,
    ) -> PyResult<String> {
        let params = RunParams {
            min_confidence,
            max_results,
            region: region.map(|(x, y, w, h)| PixelRect { x, y, w, h }),
        };
        let result = py
            .detach(|| {
                let image = ImageInput::new(&pixels, width, height, channels)?;
                self.0.run(&image, &params)
            })
            .map_err(raise)?;
        Ok(serde_json::to_string(&result).expect("results serialize"))
    }
}

#[pyclass(frozen, module = "syrup._native")]
struct NativeSession(Mutex<Session>);

#[pymethods]
impl NativeSession {
    #[pyo3(signature = (pixels, width, height, channels, min_confidence=None, max_results=None, region=None))]
    #[allow(clippy::too_many_arguments)]
    fn update(
        &self,
        py: Python<'_>,
        pixels: PyBackedBytes,
        width: u32,
        height: u32,
        channels: u32,
        min_confidence: Option<f32>,
        max_results: Option<u32>,
        region: Option<(u32, u32, u32, u32)>,
    ) -> PyResult<String> {
        let params = RunParams {
            min_confidence,
            max_results,
            region: region.map(|(x, y, w, h)| PixelRect { x, y, w, h }),
        };
        let result = py
            .detach(|| {
                let image = ImageInput::new(&pixels, width, height, channels)?;
                let mut session = self.0.lock().unwrap_or_else(|e| e.into_inner());
                session.update(&image, &params)
            })
            .map_err(raise)?;
        Ok(serde_json::to_string(&result).expect("results serialize"))
    }
}

#[pyfunction]
fn resolve(name: &str) -> PyResult<NativeOperation> {
    runtime()?.resolve(name).map(NativeOperation).map_err(raise)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Spec {
    find: String,
    color: Option<String>,
    region: Option<RegionArg>,
    order: Option<String>,
    limit: Option<u32>,
    min_area_pct: Option<u32>,
    max_area_pct: Option<u32>,
    measure: Option<String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum RegionArg {
    Named(String),
    Fractions([f64; 4]),
}

fn intent_from(spec: Spec) -> Result<Intent, SyrupError> {
    let malformed = |reason: String| SyrupError::new(Stage::Resolve, ErrorKind::Malformed, reason);
    let measure = match spec.measure {
        None => None,
        Some(word) => Some(
            catalog::Quantity::named(&word)
                .ok_or_else(|| malformed(format!("{word:?} is not a quantity Syrup measures")))?,
        ),
    };
    // "image" names the searched image or region, which only measurements use.
    let target = match spec.find.as_str() {
        "image" if measure.is_some() => Some(catalog::Target::Image),
        noun => catalog::target_named(noun),
    }
    .ok_or_else(|| {
        SyrupError::new(
            Stage::Resolve,
            ErrorKind::Unsupported,
            format!("nothing in the catalog finds {:?}", spec.find),
        )
        .with_hint(format!("known targets: {}", catalog::known_targets()))
    })?;
    let color = match spec.color {
        None => None,
        Some(name) => Some(
            catalog::color_named(&name)
                .ok_or_else(|| malformed(format!("{name:?} is not a colour Syrup knows")))?,
        ),
    };
    let region = match spec.region {
        None => None,
        Some(RegionArg::Named(name)) => Some(
            region_named(&name)
                .ok_or_else(|| malformed(format!("{name:?} is not a region name")))?,
        ),
        Some(RegionArg::Fractions(f)) => {
            let ratio = |v: f64| {
                Ratio::from_f64(v)
                    .ok_or_else(|| malformed(format!("region fraction {v} is not in [0, 1]")))
            };
            let rect = NormRect::new(ratio(f[0])?, ratio(f[1])?, ratio(f[2])?, ratio(f[3])?)
                .ok_or_else(|| malformed(format!("region {f:?} is empty or leaves the image")))?;
            Some(RegionSpec::Fixed { rect })
        }
    };
    let order = match spec.order {
        None => OrderKey::ConfidenceDesc,
        Some(word) => OrderKey::parse(&word)
            .ok_or_else(|| malformed(format!("{word:?} is not an ordering")))?,
    };
    Ok(Intent {
        target,
        color,
        region,
        order,
        limit: spec.limit,
        min_area_pct: spec.min_area_pct,
        max_area_pct: spec.max_area_pct,
        measure,
        track: false,
    })
}

#[pyfunction]
fn define(name: &str, spec: &str) -> PyResult<NativeOperation> {
    let spec: Spec = serde_json::from_str(spec).map_err(|e| {
        raise(
            SyrupError::new(Stage::Resolve, ErrorKind::Malformed, e.to_string())
                .for_operation(name),
        )
    })?;
    let intent = intent_from(spec).map_err(|e| raise(e.for_operation(name)))?;
    runtime()?
        .define(name, intent)
        .map(NativeOperation)
        .map_err(raise)
}

/// A detector written in Python, called by generated modules through the
/// host like any other provider.
struct PythonDetector {
    name: &'static str,
    /// What provenance calls the detector, e.g. the model it runs.
    provider: &'static str,
    model_sha256: Option<String>,
    detect: Py<PyAny>,
}

type Found = (f32, f32, f32, f32, f32, Option<String>);

impl Provider for PythonDetector {
    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            capability: self.name,
            name: self.provider,
            model_sha256: self.model_sha256.clone(),
            runtime: "python",
        }
    }

    fn detect(&self, view: &ViewRef<'_>) -> syrup_runtime::Result<Detections> {
        let pixels = view.packed();
        let found: Vec<Found> = Python::attach(|py| {
            let pixels = PyBytes::new(py, &pixels);
            self.detect
                .call1(py, (pixels, view.width, view.height, view.channels))?
                .extract(py)
        })
        .map_err(|e: PyErr| {
            SyrupError::new(
                Stage::Execute,
                ErrorKind::ProviderFailed,
                format!("the {} detector failed: {e}", self.name),
            )
        })?;
        let with_text = found.iter().any(|f| f.5.is_some());
        let mut out = Detections::default();
        for (x, y, w, h, score, text) in found {
            out.boxes.push(SyrupDetection {
                x,
                y,
                w,
                h,
                score,
                value: 0.0,
                n_keypoints: 0,
                keypoints: [0.0; 10],
                payload: 0,
            });
            if with_text {
                out.texts.push(text.unwrap_or_default());
            }
        }
        Ok(out)
    }
}

#[pyfunction]
#[pyo3(signature = (singular, plural, min_confidence, detect, provider="python", model_sha256=None))]
fn add_target(
    singular: &str,
    plural: &str,
    min_confidence: f32,
    detect: Py<PyAny>,
    provider: &str,
    model_sha256: Option<String>,
) -> PyResult<()> {
    let leak = |s: &str| -> &'static str { Box::leak(s.to_string().into_boxed_str()) };
    let detector = Arc::new(PythonDetector {
        name: leak(singular),
        provider: leak(provider),
        model_sha256,
        detect,
    });
    syrup_runtime::custom::add_target(singular, plural, min_confidence, detector)
        .map(|_| ())
        .map_err(raise)
}

#[pyfunction]
fn decode_image(py: Python<'_>, path: PathBuf) -> PyResult<(Py<PyBytes>, u32, u32, u32)> {
    let image = py.detach(|| OwnedImage::open(&path)).map_err(raise)?;
    let data = PyBytes::new(py, &image.data).unbind();
    Ok((data, image.width, image.height, image.channels))
}

#[pyfunction]
fn bundle(py: Python<'_>, path: PathBuf, names: Vec<String>) -> PyResult<String> {
    let runtime = runtime()?;
    let names: Vec<&str> = names.iter().map(String::as_str).collect();
    let index = py.detach(|| runtime.bundle(&path, &names)).map_err(raise)?;
    Ok(serde_json::to_string(&index).expect("indexes serialize"))
}

/// The words `define` accepts, from the catalog as it is now, including
/// targets added in this process.
#[pyfunction]
fn vocabulary() -> String {
    let targets: Vec<&str> = catalog::all()
        .into_iter()
        .filter_map(|e| e.singular.first().copied())
        .chain(["image"])
        .collect();
    serde_json::json!({
        "targets": targets,
        "colors": catalog::COLORS.iter().map(|c| c.names[0]).collect::<Vec<_>>(),
        "regions": region_names(),
        "orders": ["confidence", "size", "area_asc", "left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top"],
        "quantities": catalog::Quantity::ALL.map(catalog::Quantity::name),
        "grammar": syrup_runtime::intent::grammar_summary(),
    })
    .to_string()
}

#[pyfunction]
fn list_windows(py: Python<'_>) -> PyResult<Vec<String>> {
    py.detach(WindowCapture::windows).map_err(raise)
}

/// Pixels, width, height and channels, as `syrup.Image` takes them.
type Frame = (Py<PyBytes>, u32, u32, u32);

/// A live window, captured frame by frame.
#[pyclass(name = "Window")]
struct PyWindow {
    title: String,
    capture: Mutex<WindowCapture>,
}

#[pymethods]
impl PyWindow {
    #[new]
    fn new(py: Python<'_>, query: &str) -> PyResult<PyWindow> {
        let capture = py.detach(|| WindowCapture::new(query)).map_err(raise)?;
        Ok(PyWindow {
            title: capture.title().to_string(),
            capture: Mutex::new(capture),
        })
    }

    #[getter]
    fn title(&self) -> &str {
        &self.title
    }

    /// The next frame as (pixels, width, height, channels), or None once
    /// the window has closed.
    fn capture(&self, py: Python<'_>) -> PyResult<Option<Frame>> {
        let frame = py
            .detach(|| {
                let mut capture = self.capture.lock().unwrap_or_else(|e| e.into_inner());
                capture.next_frame()
            })
            .map_err(raise)?;
        Ok(frame.map(|frame| {
            let data = PyBytes::new(py, &frame.data).unbind();
            (data, frame.width, frame.height, frame.channels)
        }))
    }
}

#[pyfunction]
fn cache_dir() -> PyResult<PathBuf> {
    Ok(runtime()?.store().root().to_path_buf())
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("NativeError", m.py().get_type::<NativeError>())?;
    m.add_class::<NativeOperation>()?;
    m.add_class::<NativeSession>()?;
    m.add_function(wrap_pyfunction!(resolve, m)?)?;
    m.add_function(wrap_pyfunction!(define, m)?)?;
    m.add_function(wrap_pyfunction!(add_target, m)?)?;
    m.add_function(wrap_pyfunction!(decode_image, m)?)?;
    m.add_function(wrap_pyfunction!(bundle, m)?)?;
    m.add_function(wrap_pyfunction!(vocabulary, m)?)?;
    m.add_function(wrap_pyfunction!(list_windows, m)?)?;
    m.add_class::<PyWindow>()?;
    m.add_function(wrap_pyfunction!(cache_dir, m)?)?;
    Ok(())
}
