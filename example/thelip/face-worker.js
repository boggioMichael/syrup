/**
 * The face finder's Web Worker: BlazeFace (faces.js, prepended by build.py)
 * on 128x128 pictures the page letterboxes a frame into.
 *
 *   -> { type: "init", model: ArrayBuffer }            (blazeface.bin)
 *   <- { type: "ready", ms } | { type: "failed", error }
 *   -> { type: "detect", id, rgba: Uint8ClampedArray }  (128*128*4)
 *   <- { type: "faces", id, faces: [{score, box, keypoints}], ms }   (fractions of the 128x128 picture)
 */
let detector = null;
onmessage = (e) => {
  const m = e.data;
  if (m.type === "init") {
    try {
      detector = new FaceDetector(loadFaceModel(m.model));
      const t0 = performance.now();
      detector.detect(new Uint8ClampedArray(FACE_INPUT * FACE_INPUT * 4));
      postMessage({ type: "ready", ms: performance.now() - t0 });
    } catch (err) { postMessage({ type: "failed", error: String(err && err.message || err) }); }
  } else if (m.type === "detect" && detector) {
    const t0 = performance.now();
    const faces = detector.detect(m.rgba);
    postMessage({ type: "faces", id: m.id, faces, ms: performance.now() - t0 });
  }
};
