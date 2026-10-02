# syrup-cv

Computer-vision operations you name instead of write.

```python
from syrup.ops import find_largest_face, find_red_bars_in_bottom_third

faces = find_largest_face("photo.jpg")
for face in faces:
    print(face.box, face.confidence)
```

The name is parsed into a plan, compiled to a small native module the
first time it runs, checked against a reference implementation, and reused
afterwards. Names Syrup cannot honour raise `syrup.IntentError` at import.

Compiling needs `rustc` 1.82 or newer. Machines without it can run
operations prepared elsewhere:

```python
syrup.bundle("ops-bundle", "find_largest_face", "find_words")
# then, on the target machine: SYRUP_MODE=frozen SYRUP_CACHE_DIR=ops-bundle
```

Detectors from other libraries plug in with `syrup.add_target`. The grammar,
result contract and failure classes are in
[crates/syrup-runtime/docs/contract.md](https://github.com/boggioMichael/syrup/blob/main/crates/syrup-runtime/docs/contract.md).

## Install

```sh
pip install ./python            # builds the extension with maturin
pip install "./python[test]"    # and what the tests need
pytest python/tests
```

## The in-process intents (`syrup.find_face`, `syrup.legacy`)

Names on the package itself — `syrup.find_face`, `syrup.find_eyes`,
`syrup.measure_red_bar`, `syrup.find_<name>_icon` — are the core library's
in-process intents (`syrup::intent`: cascade faces, eyes and profile faces,
icons by template, bars, blobs, text, motion), reached through its C API
(`src/capi.rs`). They are a separate vocabulary with their own results,
`Detection(value, confidence, reliability, failure_reason)`, and need only
the core library: `cargo build --release` produces
`target/release/libsyrup.so` (`.dylib`, `syrup.dll`), which `syrup.legacy`
finds from the checkout or through `SYRUP_LIBRARY`.

```sh
cargo build --release
python3 python/test_legacy.py
```
