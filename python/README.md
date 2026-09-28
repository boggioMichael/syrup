# syrup for Python

```python
import syrup
from PIL import Image

image = Image.open("photo.jpg")
for face in syrup.find_face(image).value:
    print(face.bounds, face.score)
print(syrup.find_eyes(image).value)
print(syrup.measure_red_bar(image).value)      # a percentage, or None with .failure_reason
```

Any attribute named like an intent — `find_*`, `track_*`, `count_*`,
`measure_*`, `read_*` — is resolved through the native library the first
time it is used. Outside the vocabulary you get an `IntentError` whose
message lists it. `syrup.find_face.source` is the Rust the library wrote for
the name; `syrup.find_face.compile()` builds it into its own shared library.

Images are PIL images or numpy `HxWx3`/`HxWx4` `uint8` arrays. Results are
`Detection(value, confidence, reliability, failure_reason)`: `value` is a
list of `Match(bounds, score, centre, id)`, a float, or a string — or `None`
when the search could not run, with the reason.

## Setup

The package is a thin `ctypes` layer over the library's C API (`src/capi.rs`),
so it needs the native library: `cargo build --release` in the repository
produces `target/release/libsyrup.so` (`.dylib`, `syrup.dll`), which the
package finds on its own from the checkout, or through `SYRUP_LIBRARY`.

```sh
cargo build --release
cd python && python3 -m pytest        # or: python3 test_syrup.py
```
