# Missing-Person Face Extraction — AI_VMS-ready usecase

Detects every person in a video with YOLOv8, then extracts a usable, verified
face crop from each one using InsightFace `buffalo_l` — the candidate pool
for a downstream missing-person face-matching/search step. **Single-toolkit
design**: InsightFace `buffalo_l` is the only face model — no MediaPipe, no
second recognition model. All 5 of its bundled sub-models are used (detection,
2D/3D landmarks via `face.pose`, age/gender, and the ArcFace recognition
embedding), not just detection.

It's structured to match **AI_VMS**'s real 3-stage convention (AI Orchestrator
→ Business Logic Orchestrator → Event Manager) so the usecase code can later
be copied into the real AI_VMS repository with no logic changes — only file
*locations* change. `run_local.py` exists purely to run and test this
locally, without RabbitMQ/Docker/Postgres; it does not get ported.

---

## 1. The pipeline, end to end

```
 video frame
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│ STAGE 1 — AI_Orchestration/Detection/YOLO/inference.py   │
│ YOLOv8 + ByteTrack finds + tracks every person            │
│ → [{"xyxy":[...], "class_name":"person", "track_id":N}]   │
│ (track_id is CHEAP but UNSTABLE -- resets on occlusion)    │
└─────────────────────────────────────────────────────────┘
     │  batch["inference_results"]
     ▼
┌─────────────────────────────────────────────────────────┐
│ STAGE 2 — BL_Orchestration/.../missing_person_face_      │
│           extraction.py                                  │
│ for each TRACKED person box:                              │
│   crop → (upscale if small/distant) → InsightFace         │
│   buffalo_l (ALL 5 sub-models) → quality gate               │
│   → score (confidence + sharpness + size +                  │
│             frontal-pose from buffalo_l's own face.pose)     │
│   → face-embedding ReID: track_id -> global_person_id        │
│     (cosine-similarity match against known identities,        │
│      recovers identity after a tracker ID reset)               │
│   → keep only if better than this global_person_id's best      │
│ draws red (person) / green (qualifying face, labeled "ID n")    │
│ overwrites output/best_faces/person_<camera>_<id>.{jpg,npy,json} │
│ → batch["bl_results"] = {"faces":[...], "alert": bool}            │
└─────────────────────────────────────────────────────────┘
     │  batch["bl_results"]
     ▼
┌─────────────────────────────────────────────────────────┐
│ STAGE 3 — Event_Manager/.../missing_person_face_          │
│           extraction.py                                   │
│ if alert and not a duplicate (10s cooldown):               │
│   dump the last ~10s of frames to an evidence .mp4 clip    │
└─────────────────────────────────────────────────────────┘
     │
     ▼
 output/annotated_video.mp4, output/best_faces/person_<camera>_<id>.{jpg,npy,json}, output/evidence/*.mp4
```

Each stage is a plain Python module with **one mandatory entrypoint function**
that gets the whole frame's context passed in as a `batch` dict and mutates
it in place — this is exactly the contract AI_VMS's real orchestrators use to
dynamically load and call usecase code, which is why the same three files can
be copied into AI_VMS almost unchanged (see §5).

---

## 2. Directory structure

```
person_face_pipeline/
├── run_local.py                                     # wires the 3 stages together, reads ONE local video
├── search_multi_camera.py                            # targeted search for one specific person, N cameras in parallel, one combined live window
│
├── AI_Orchestration/                                 # mirrors AI_VMS's AI_Models/AI_Orchestration
│   ├── Detection/YOLO/
│   │   ├── inference.py                              # STAGE 1: person detection + tracking
│   │   └── yolov8n.pt                                # model weights
│   ├── config/config.ini                             # DEVICE, PERSON_CONF, PERSON_IMGSZ, TRACKER
│   ├── constant/constants.py                         # config key names
│   └── services/config_reader.py                     # local stand-in ConfigManager
│
├── BL_Orchestration/                                  # mirrors AI_VMS's Business_Logics/BL_Orchestration
│   ├── Usecases_Core_logics/MISSING_PERSON_FACE_EXTRACTION/
│   │   └── missing_person_face_extraction.py          # STAGE 2: face extraction, scoring, ReID
│   ├── config/config.ini                              # face-detection + ReID thresholds
│   ├── constant/constants.py
│   └── services/config_reader.py                      # local stand-in ConfigManager
│
├── Event_Manager/                                      # mirrors AI_VMS's Event_Manager
│   └── Usecases_Event_logics/MISSING_PERSON_FACE_EXTRACTION/
│       └── missing_person_face_extraction.py           # STAGE 3: alerting + evidence clips
│
├── config/
│   └── search_cameras.ini                            # camera list + settings for search_multi_camera.py
│
├── input/                                              # source videos
├── output/                                             # annotated_video.mp4, best_faces/, evidence/
└── requirements.txt
```

---

## 3. File-by-file walkthrough

### `run_local.py` — the local wiring script

This is the only file that doesn't port into AI_VMS; it stands in for AI_VMS's
real `orchestrator.py` / `bl_orchestrator.py` / `event_orchestrator.py`
(which normally run as 3 separate RabbitMQ-fed containers).

- **`_ensure_gpu_libs_on_path()`** (lines 27-50) — a local-machine-only fix.
  PyTorch/onnxruntime ship their own CUDA/cuDNN `.so` files inside pip
  packages, but nothing puts that folder on `LD_LIBRARY_PATH`, and the
  dynamic linker only reads that variable once, at process startup. This
  function finds those library folders, sets the env var, and restarts the
  process (`os.execv`) so the fix actually takes effect — guarded by a flag
  so it only happens once.
- **`_load_stage_module()`** (lines 75-87) — loads one stage's `.py` file
  from an explicit path (`importlib.util.spec_from_file_location`), the same
  mechanism AI_VMS's real orchestrators use to dynamically load usecase code
  by folder-name convention. It's wrapped with sys.path bookkeeping because
  all three stages independently define their own `constant`/`services`
  packages — fine in real AI_VMS since each stage runs in its own container,
  but a naming collision if all three get imported into one process like
  here. This function inserts only the current stage's folder onto
  `sys.path`, loads it, then removes that folder and purges the cached
  `constant`/`services` modules so the next stage starts clean.
- **`main()`** (lines 90-181) — parses CLI args, loads the 3 stage modules,
  opens the video, and for every frame:
  1. `ai_module.run_inference([frame], {})` → person boxes
  2. builds a `batch` dict shaped like AI_VMS's real message (`camera_metadata`,
     `frame_metadata`, `inference_results`)
  3. `bl_module.run_logic(batch)` → extracts faces, annotates the frame, sets `batch["bl_results"]`
  4. `event_module.run_event_logic(batch, "MISSING_PERSON_FACE_EXTRACTION")` → handles alerting/evidence
  5. decodes the (now jpg-bytes-encoded) annotated frame back to an image, writes it to the output video and/or preview window

### `AI_Orchestration/Detection/YOLO/inference.py` — Stage 1: person detection + tracking

Matches AI_VMS's real `Detection/YOLO/inference.py` structure exactly (a
`YOLOModel` class + module-level `run_inference` entrypoint), plus two
additions: **`imgsz` support** and **tracking**.

- **`YOLOModel.__init__`/`load_model`** — loads `yolov8n.pt` onto the
  requested device, falling back to CPU if CUDA isn't available or loading
  fails for any reason.
- **`YOLOModel.run()`** — the actual per-frame work. Reads `PERSON_CONF` (min
  confidence) and **`PERSON_IMGSZ`** from config; `imgsz` matters specifically
  for catching **distant/small people**: YOLO internally resizes the frame to
  this resolution before detecting anything — the default 640px shrinks
  far-away people to near-invisible before detection even runs. Raising it to
  1280 (the default here) catches them, at some added inference cost. When
  the caller passes `metadata={"tracking": True}` (as `run_local.py` does),
  it calls `model.track(..., persist=True, tracker="bytetrack.yaml")` instead
  of `model.predict(...)` — `persist=True` keeps ByteTrack's internal state
  alive across calls so the same person keeps the same `track_id` from frame
  to frame, which Stage 2 needs to pick one best face per person over time
  rather than treating every frame's detection as a stranger.
- **`resolve_device()`** — `DEVICE` config value: a digit string (e.g. `"0"`)
  resolves to `cuda:0` if available else `cpu`; any other string (e.g.
  `"cpu"`) is returned as-is. Same convention AI_VMS's real config uses.
- **`run_inference()`** — the mandatory entrypoint. Keeps one `YOLOModel`
  instance per device in `_model_instances` so the model (and its tracker
  state) persists across calls instead of resetting every frame.

### `BL_Orchestration/.../missing_person_face_extraction.py` — Stage 2: scored extraction + ReID

The core business logic — and the only file besides Stage 1 that touches a
model. Reads config once at import time: `DET_SIZE`, `MIN_FACE_PX`,
`FACE_DET_CONF`, `BLUR_MIN`, `UPSCALE_BELOW`, `MAX_UPSCALE`, `BEST_FACES_DIR`,
`REID_THRESHOLD`, the 4 score weights (`SCORE_W_*`), and 2 normalization
reference points (`SHARP_REF_MAX`, `SIZE_REF_MAX`).

- **`_frontal_score(pose)`** — `pose` is `buffalo_l`'s own `[pitch, yaw, roll]`
  in degrees, already computed by its `landmark_3d_68` sub-model (a 3D-to-3D
  fit against a mean face shape — verified directly from InsightFace's source:
  `insightface/model_zoo/landmark.py`). No separate pose-estimation library or
  `solvePnP` call is needed; `|pitch| + |yaw|` (roll/in-plane tilt is ignored
  — a tilted-but-frontal face is still usable) is converted to a 0-1
  "frontal-ness" score. There's no eye-openness term: `buffalo_l`'s 106-point
  landmark index layout for eye contours isn't documented anywhere in the
  package, and guessing indices wrong would silently produce a garbage signal
  — dropped rather than risk that.
- **`FaceExtractor.extract()`** — given one person's cropped image:
  1. **Upscale if small**: if the crop's smaller side is under
     `UPSCALE_BELOW` (default 220px), it's upsampled (capped at
     `MAX_UPSCALE`, default 4x) via `cv2.INTER_CUBIC` before detection —
     recovers genuinely detectable faces from far-away people.
  2. **Detect**: `face_app.get(detect_crop)` — one call runs all 5 `buffalo_l`
     sub-models, returning bbox, 5-point landmarks, detection confidence,
     `pose`, `age`, `gender`, and the 512-d ArcFace embedding
     (`normed_embedding`).
  3. **Quality-gate each face** (not just the best one — a box can contain
     2+ overlapping people): confidence ≥ `FACE_DET_CONF`, ≥3 landmarks
     present, `min(width,height)` ≥ `MIN_FACE_PX`, sharpness ≥ `BLUR_MIN`.
  4. **Score**: `score = W_CONF·conf + W_SHARP·sharpness + W_SIZE·size + W_FRONTAL·frontal`.
  5. **Coordinate correction**: bbox divided back by `scale` since detection
     ran on the (possibly) upscaled crop.
- **Face-embedding ReID** (`_resolve_global_id`) — the tracker-to-identity
  layer. `track_id` from ByteTrack is cheap but unstable: it resets whenever
  a person is occluded or leaves and re-enters frame. This function maps it
  to a stable `global_person_id`:
  - First time a `track_id` is seen: compare its face embedding (cosine
    similarity — both vectors are already L2-normalized, so this is a plain
    dot product) against every existing `global_person_id`'s stored
    embedding, per camera.
  - Best match ≥ `REID_THRESHOLD` (default 0.45) → merge: map this
    `track_id` to that *existing* identity (the tracker lost and reassigned
    someone already seen; ReID recovers them).
  - No good match → mint a new `global_person_id`.
  - Once resolved, the mapping is cached — subsequent frames with the same
    `track_id` skip straight to its `global_person_id`.
- **`_update_best_face()`** — per-`global_person_id` deduplication (not raw
  `track_id` anymore). A new candidate is only written to disk if it beats
  the stored score, overwriting `output/best_faces/person_<camera_id>_<global_id>.jpg`,
  its `.npy` embedding, and a `.json` with `age`, `gender`, `det_score`,
  `score`, and a timestamp. `camera_id` is part of the filename because
  `global_person_id` is only unique *within* one camera's own gallery — two
  different cameras can both produce id `0`.
- **`run_logic()`** — the mandatory entrypoint:
  1. Decodes `frame_metadata["frame"]` if it arrived as encoded bytes.
  2. Filters `inference_results` down to `class_name == "person"` boxes
     (each carrying a `track_id` from Stage 1's tracker).
  3. Draws all person boxes in **red**.
  4. For each person box: crops it, calls `extract()`, resolves each
     candidate's `global_person_id` via ReID, draws it in **green** labeled
     `"ID <n>"`, calls `_update_best_face()`, and appends a summary to the
     result list.
  5. Re-encodes the now-annotated frame back to jpg bytes (matching how real
     AI_VMS messages travel over RabbitMQ/msgpack).
  6. Sets `batch["bl_results"] = {"faces": [...], "alert": len(faces) > 0}`.

### `Event_Manager/.../missing_person_face_extraction.py` — Stage 3: alerting + evidence

- **`_is_duplicate()`** (lines 44-50) — a simple per-camera cooldown
  (`ALERT_COOLDOWN_SECONDS`, default 10s) so a continuously-visible face
  doesn't trigger a new evidence clip on every single frame.
- **`_get_buffer()` / `_save_clip()`** (lines 53-74) — each camera gets its
  own rolling `deque` sized to hold the last `CLIP_SECONDS` (default 10)
  worth of frames. Every frame gets appended (line 90) regardless of whether
  it triggered anything; when an alert fires and isn't a duplicate, the
  entire buffer is dumped to an `.mp4` in `output/evidence/` — a clip
  centered on the moment a face was found, not just a single still frame.
- **`run_event_logic()`** (lines 77-106) — the mandatory entrypoint: buffers
  the frame, returns early if no alert or if it's a duplicate, otherwise logs
  and saves the clip.

  ⚠️ This rolling-buffer approach is a **local-only stand-in**. Real AI_VMS's
  Event orchestrator already passes several recent frames per call (a real
  `batch`, plural), so its usecases build evidence clips directly from that
  — see §5 for what to swap in when porting.

### The two `config_reader.py` files

Both are minimal, local-only stand-ins for AI_VMS's real `services/config_reader.py`,
implementing the same interface real usecase code calls:
`cfg.get_str/get_int/get_float(key, fallback)` (used in `AI_Orchestration`)
and `cfg.get_value_config(section, key)` (used in `BL_Orchestration`, matching
how real per-usecase config sections work, e.g. `[MISSING_PERSON_FACE_EXTRACTION]`).
They read from each stage's own `config.ini`, with environment variables
taking precedence — same override rule AI_VMS's real `ConfigManager` uses.
**Neither file is meant to be copied into AI_VMS** — it already has its own.

### The two `constants.py` files

Just named string constants for config keys (e.g. `Constants.FACE_DET_CONF_KEY
= "FACE_DET_CONF"`), so the rest of the code never hardcodes a raw config-key
string. Only the keys this usecase actually reads — AI_VMS's real
`constants.py` files have many more, unrelated to this usecase.

---

## 4. The `batch` dict — what flows between stages

```python
batch = {
    "camera_metadata": {"camera_id": "local-cam-1"},
    "frame_metadata": {
        "frame": <jpg bytes or raw np.ndarray>,   # image, mutated in place by each stage
        "fps": 25.0,
    },
    "inference_results": [                         # one list per frame (here: always 1 frame)
        [
            {"xyxy": [x1, y1, x2, y2], "confidence": 0.93, "class_name": "person",
             "id": 0, "track_id": 7},               # track_id from ByteTrack (Stage 1)
            ...
        ]
    ],
    # added by Stage 2 (best face itself is written to output/best_faces/, not embedded here):
    "bl_results": {
        "faces": [
            {"global_person_id": 3, "track_id": 7, "bbox": [x1, y1, x2, y2],
             "det_score": 0.87, "score": 0.7421},
            ...
        ],
        "alert": True,
    },
}
```

---

## 5. Config reference

**`AI_Orchestration/config/config.ini`**

| Key | Default | Meaning |
|---|---|---|
| `DEVICE` | `0` | GPU index digit (→ `cuda:0`, falls back to CPU) or the literal `cpu` |
| `PERSON_CONF` | `0.25` | Min YOLOv8 confidence to count as a person |
| `PERSON_IMGSZ` | `1280` | YOLO's internal input resolution — higher catches smaller/farther people |
| `TRACKER` | `bytetrack.yaml` | Ultralytics' bundled tracker config, used when `metadata={"tracking": True}` |

**`BL_Orchestration/config/config.ini`** (section `[MISSING_PERSON_FACE_EXTRACTION]`)

| Key | Default | Meaning |
|---|---|---|
| `DEVICE` | `auto` | `auto` picks GPU if available, else CPU |
| `DET_SIZE_W` / `DET_SIZE_H` | `640` / `640` | InsightFace's own input resolution |
| `MIN_FACE_PX` | `28` | Minimum face box side, in pixels |
| `FACE_DET_CONF` | `0.40` | Minimum InsightFace detection confidence |
| `BLUR_MIN` | `30.0` | Minimum Laplacian-variance sharpness (below this = rejected as blurry) |
| `UPSCALE_BELOW` | `220` | Person crops smaller than this (px, shorter side) get upsampled before face detection |
| `MAX_UPSCALE` | `4.0` | Cap on how much a tiny/distant crop gets upsampled |
| `BEST_FACES_DIR` | `output/best_faces` | Where the single best face per global_person_id is saved |
| `REID_THRESHOLD` | `0.45` | Cosine-similarity cutoff (ArcFace embeddings) to merge a track_id into an existing identity instead of minting a new one |
| `SCORE_W_CONF` | `0.35` | Weight: InsightFace detection confidence |
| `SCORE_W_SHARP` | `0.30` | Weight: sharpness (Laplacian variance) |
| `SCORE_W_SIZE` | `0.15` | Weight: face size in pixels |
| `SCORE_W_FRONTAL` | `0.20` | Weight: head-pose frontal-ness (from buffalo_l's own `face.pose`) |
| `SHARP_REF_MAX` | `400.0` | Laplacian variance considered "very sharp" (normalization reference) |
| `SIZE_REF_MAX` | `200.0` | Face `min(w,h)` in px considered "large" (normalization reference) |

Loosen the quality-gate thresholds (`FACE_DET_CONF`, `MIN_FACE_PX`,
`BLUR_MIN`) for more recall (catches more faces, risks some false
positives); tighten them for fewer, higher-confidence crops. Adjust the
`SCORE_W_*` weights to change what "best" means. Raise `REID_THRESHOLD` for
fewer identity merges (more distinct IDs, less risk of merging two similar-
looking people); lower it for more merges (better at recovering identity
across occlusion, more risk of a false merge).

---

## 6. Running it locally

```bash
source /home/gyana/jupyter_env/bin/activate
pip install -r requirements.txt
python run_local.py --video input/video.mp4 --output output/annotated_video.mp4
```

Options: `--every N` (process every Nth frame), `--no-display` (headless).

Outputs:
- `output/annotated_video.mp4` — full video, red = person, green = qualifying face this frame, labeled with its `global_person_id`
- `output/best_faces/person_<camera_id>_<global_id>.jpg` + `.npy` + `.json` —
  the single best-scoring face crop seen so far for each ReID-resolved
  person, its 512-d ArcFace embedding (ready for cosine-similarity matching
  against a watchlist), and metadata (age, gender, det_score, score)
- `output/evidence/` — a ~10-second `.mp4` clip per detection event

---

## 7. Missing-person search across multiple cameras

`search_multi_camera.py` searches any number of camera feeds in parallel
(one Python thread per camera) for one specific missing person, given a
reference photo. Driven entirely by `config/search_cameras.ini`:

```ini
[general]
MISSING_PERSON_IMAGE = input/reference/wildtrack_missing_person.jpg
CAMERA_IDS = cam1, cam2
OUTPUT_DIR = output/search_results
SAVE_WINDOW_VIDEO = true

[camera:cam1]
INPUT_VIDEO = Wildtrack/cam1.mp4

[camera:cam2]
INPUT_VIDEO = Wildtrack/cam2.mp4
```

```bash
source /home/gyana/jupyter_env/bin/activate
python search_multi_camera.py
# or: python search_multi_camera.py --config config/my_search_cameras.ini
```

Add/remove cameras by editing `CAMERA_IDS` and adding/removing a matching
`[camera:<id>]` section — no code changes needed.

All cameras render into **one combined window**: the live cam1/cam2 grid on
top, the missing-person reference photo and a status/alert panel below it.
The alert panel turns pink with "PERSON FOUND" the instant any camera gets a
candidate; a terminal `[y/n]` prompt then either stops every camera (`y`,
also saves `MATCH_CONFIRMED_<camera_id>.jpg`) or rejects that person and
keeps searching (`n`). With `SAVE_WINDOW_VIDEO = true`, the whole window is
also recorded to `OUTPUT_DIR/window_output.mp4` for replaying the run later.

**Why one camera per thread needs care:** person tracking (ByteTrack,
`persist=True`) keeps its motion-history state *inside* the YOLO model
object. If two unrelated video streams shared one model instance, their
frames would interleave through the same tracker, corrupting `track_id`s for
**both** cameras. Fixed in `AI_Orchestration/Detection/YOLO/inference.py`:
`run_inference()` now caches model instances by `(device, tracker_key)`
instead of just `device`, and `search_multi_camera.py` passes each camera's
own `camera_id` as `tracker_key` — so every camera gets its own detector +
tracker, never sharing tracking state with another camera.

What's safely **shared** across all camera threads instead: **InsightFace
(`buffalo_l`)** — one loaded model instance for all cameras. It has no state
between calls (just detect+embed per crop), and every call is wrapped in a
lock (`FaceMatcher`, defined in `search_multi_camera.py`), so concurrent
cameras simply queue for GPU time rather than needing a second copy of the
model in memory.

Qt/OpenCV's highgui backend is **not** thread-safe, so only the main thread
ever calls `cv2.imshow`/`cv2.waitKey` — camera worker threads just write
their latest annotated frame into a shared dict for the main thread to
display.

---

## 8. Porting into AI_VMS

| Here | Goes to in AI_VMS |
|---|---|
| `AI_Orchestration/Detection/YOLO/inference.py` | merge into `AI_Models/AI_Orchestration/Detection/YOLO/inference.py` (adds `imgsz` support + tracking via `metadata={"tracking": True}` — that model already exists in AI_VMS, don't create a new folder) |
| `AI_Orchestration/Detection/YOLO/yolov8n.pt` | already present in AI_VMS — don't copy |
| `AI_Orchestration/config/config.ini`, `constant/constants.py` | merge `PERSON_CONF`/`PERSON_IMGSZ`/`TRACKER` keys into AI_VMS's real files |
| `AI_Orchestration/services/config_reader.py` | **do not copy** |
| `BL_Orchestration/Usecases_Core_logics/MISSING_PERSON_FACE_EXTRACTION/missing_person_face_extraction.py` | same path under `Business_Logics/BL_Orchestration/` |
| `BL_Orchestration/config/config.ini`, `constant/constants.py` | merge into AI_VMS's real files |
| `BL_Orchestration/services/config_reader.py` | **do not copy** |
| `Event_Manager/Usecases_Event_logics/MISSING_PERSON_FACE_EXTRACTION/missing_person_face_extraction.py` | same path in AI_VMS, **but** replace the local cooldown/rolling-buffer with real services (see below) |
| `run_local.py` | stays here — not part of AI_VMS |

**When porting the Event stage**, replace:
- `_is_duplicate(camera_id)` → `services.event_duplication_handler.EventDuplicationHandler().is_duplicate(...)`
- `_save_clip(...)` → build the evidence clip from the `batch` **list** AI_VMS's real Event orchestrator already passes (several recent frames per call) instead of keeping a local rolling buffer — see `LOITERING_DETECTION`'s real event module for the pattern
- add `services.websocket_client.ws_client.send_event(payload)` for live UI updates
- add `services.db_service.db_service.process_and_save_events(...)` to persist the alert

Also: register `MISSING_PERSON_FACE_EXTRACTION` in AI_VMS's `usecases` table
(same as any other usecase), add `insightface`/`onnxruntime` to
`Business_Logics/BL_Orchestration/requirements.txt` (no `mediapipe` needed —
this usecase is InsightFace-only), and make sure the target camera's
Detection model has `class_names = person`, **`tracking = true` (required,
not just recommended — the face-embedding ReID needs a `track_id` to key
off of, even though it recovers gracefully from that ID resetting)**.

Note: the ReID gallery (`_gallery`, `_track_to_global`) is in-memory,
per-container state — restarting the BL container loses all known
identities and starts fresh. For persistence across restarts (or across
multiple BL container replicas), this state would need to move to a shared
store (e.g. Redis or Postgres) keyed the same way: `(camera_id,
global_person_id) -> embedding`.

---

## 9. GPU / cuDNN note (this machine only)

GPU inference here crashed with `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`
until `nvidia-cudnn-cu13` was upgraded to `9.23.2.1` (matching the system's
own libcudnn). `run_local.py`'s `_ensure_gpu_libs_on_path()` handles the
required `LD_LIBRARY_PATH` fix automatically — this is local-machine-specific
and irrelevant once ported into AI_VMS's own containers.
