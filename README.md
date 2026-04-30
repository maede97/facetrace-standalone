# FaceTrace Offline

FaceTrace Offline is a standalone browser app for local face similarity comparison. It runs entirely client-side from a self-contained `index.html`; there is no server, upload, telemetry, CDN, cloud API, or remote model download.

## Run Offline

1. Open `index.html` directly in a modern browser.
2. Do not start a local server; no `localhost` connection is required or used.
3. Select one reference image, then select one or many candidate images.

The app is designed to work from a direct `file://` open. The face-api.js library and model weights are embedded directly inside `index.html`, so the browser does not need to load additional JavaScript, fetch model shards from the local filesystem, or contact the network.

## Source Layout

`index.html` is generated. For normal maintenance, edit these inputs instead:

- `src/index.template.html`: HTML shell with inline placeholders.
- `src/styles.css`: application styles.
- `src/canvas-readback-patch.js`: early Canvas 2D readback hint patch.
- `src/app.js`: application logic.
- `vendor/face-api.min.js`: vendored face-api.js runtime.
- `models/embedded-models.js`: generated JavaScript object containing local model manifests and shards; do not edit it by hand.

The unpacked files in `models/` are kept as source/audit copies of the model assets used to generate `models/embedded-models.js`.

## Build

The build uses only Python 3.9+ from the standard library. No npm install, package manager, local web server, or network access is required.

```bash
python3 tools/build.py
```

This writes two generated artifacts:

- `models/embedded-models.js`, built from the unpacked model manifests and shards in `models/`.
- `index.html`, built from `src/`, `vendor/face-api.min.js`, and the generated model bundle.

To verify that both generated files match the source inputs:

```bash
python3 tools/build.py --check
```

The build script intentionally rejects generated HTML that reintroduces external-loading tags such as `<script src>`, `<link>`, or `<iframe>`.

## Why This Build Is Unusual

The target environment is restricted: it may allow opening one local HTML file but reject `localhost`, local servers, remote URLs, CDNs, and even sibling `file://` script/model loads because file URLs can be treated as unique security origins. A normal web build that emits separate JavaScript chunks or model files is therefore less reliable for this deployment.

For that reason, the repository keeps maintainable source files, then compiles them into one large HTML artifact. The final `index.html` is intentionally big because it includes the app, face-api.js, and the model data needed for offline face comparison.

## Browser Compatibility

Use a recent Chrome, Edge, Firefox, or Safari with JavaScript, Canvas, Blob, File API, WebGL or CPU TensorFlow.js execution, `Response`, and `atob` support enabled. Very locked-down enterprise browsers must allow JavaScript execution in the opened local HTML file.

## Included Models And Library

- `index.html`: self-contained runtime application with embedded face-api.js and embedded model data.
- `vendor/face-api.min.js`: unpacked face-api.js v0.22.2 reference copy, MIT license. The local license copy is `vendor/face-api.LICENSE`.
- `models/tiny_face_detector_*`: face detection model files from the `justadudewhohacks/face-api.js` weights distribution.
- `models/face_landmark_68_*`: landmark model files from the `justadudewhohacks/face-api.js` weights distribution.
- `models/face_recognition_*`: 128D face descriptor model files from the `justadudewhohacks/face-api.js` weights distribution.
- `models/embedded-models.js`: generated local bundle of the same model manifests and shards, also embedded into `index.html`.

Review the upstream face-api.js project and model-weight terms before redistributing outside your own use case.

## Similarity Percentage

Each detected face is converted locally into a 128-number descriptor. Candidate faces are compared with the selected reference descriptor using Euclidean distance.

The user-facing percentage is:

```text
similarity = clamp((1 - euclidean_distance / 1.20) * 100, 0, 100)
```

This mapping is deterministic and monotonic: distance `0` maps to `100%`, distance `0.60` maps to `50%`, and distance `1.20` or higher maps to `0%`. Raw Euclidean distance and cosine similarity are shown only inside each result's technical details section.

Interpretation bands:

- `85-100%`: very similar
- `70-84%`: similar
- `50-69%`: possibly similar
- Below `50%`: low similarity

These thresholds are deliberately conservative and are not forensic proof.

## Privacy

All processing happens locally in your browser. No data leaves your device. The app does not upload files, call external URLs, include telemetry, or require browser permissions beyond selecting local files.

## Limitations

- Results are probabilistic and must not be used for legal, forensic, employment, access-control, or identity-verification decisions.
- Lighting, pose, blur, age differences, occlusion, image compression, and low resolution can change scores.
- The app compares visible detected faces only. If a face is not detected, no descriptor can be computed.
- If a candidate image contains multiple faces, all detected faces are scored and the closest face is used for the main result.
- If the reference image contains multiple faces, the app defaults to the largest/highest-confidence face and lets you choose a different detected reference face.

## Troubleshooting

- **Model not loaded**: confirm you are opening the generated self-contained `index.html`, not an older copy that still referenced external local scripts.
- **Canvas readback warning**: current `index.html` applies the Canvas 2D `willReadFrequently` hint before face-api.js runs. If an old browser still logs this warning, it is a performance hint rather than a correctness failure.
- **Image could not be read**: try a common browser-readable format such as JPEG, PNG, WebP, AVIF, or BMP.
- **No face detected**: use a clearer image with a larger, front-facing face and fewer occlusions.
- **Slow large batches**: the app processes images sequentially and yields between files to keep the interface responsive. Very large images and large folders can still take time.
- **No CSV download**: confirm the browser allows downloads initiated by local pages.
