"""MISSING_PERSON_FACE_EXTRACTION business logic.


Single-toolkit design: InsightFace buffalo_l is the ONLY model here (besides
the AI Orchestrator's person detector/tracker). buffalo_l bundles 5 ONNX
sub-models per call -- detection, 2D/3D landmarks, age/gender, and the
512-d ArcFace recognition embedding -- and this usecase uses all five, not
just detection:

    person box -> crop -> (upscale if small/distant) -> InsightFace buffalo_l
        -> quality gate (min size, blur, confidence)
        -> quality SCORE (confidence, sharpness, size, head-pose frontal-ness
           from buffalo_l's own landmark_3d_68 pose output -- no separate
           pose-estimation library needed)
        -> face-embedding ReID (see below)
        -> best-face-per-person update (jpg + embedding .npy + age/gender/
           score metadata .json)

Tracker + ReID
--------------
YOLOv8+ByteTrack (AI Orchestrator) gives a per-frame track_id, which is CHEAP
but UNSTABLE: it resets whenever a person is occluded or leaves and
re-enters frame. Rather than dedupe best-faces on that raw, churn-prone
track_id, this module runs a second, more durable identity layer on top of
it: face-embedding ReID.

    track_id -> global_person_id  (resolved once per track_id, then cached)

The first time a track_id is seen, its face embedding is compared (cosine
similarity) against every existing global_person_id's stored embedding. A
match above REID_THRESHOLD means the tracker lost and reassigned an ID for
someone already seen -- merge into that same global_person_id instead of
creating a new one. No match -> a new global_person_id is minted. All
best-face bookkeeping (and the folder naming: person_<global_id>.*) is keyed
by this stable global_person_id, not the raw track_id.

No front/back orientation pre-filter -- every detected person gets a face
attempt; if a face genuinely isn't visible, InsightFace simply finds nothing
for that person. A person box can also yield more than one face candidate
(two people overlapping in one box).

Overlay colors: red = person box, green = face that passed the quality gate
this frame (labeled with its global_person_id).
"""
import os
import json
import time
import logging
import threading

import cv2
import numpy as np

from services.config_reader import cfg
from constant.constants import Constants

logger = logging.getLogger("MissingPersonFaceExtraction")

CONFIG_SECTION = "MISSING_PERSON_FACE_EXTRACTION"

try:
    from insightface.app import FaceAnalysis
except ImportError:
    FaceAnalysis = None


def _cfg(key, default):
    try:
        return cfg.get_value_config(CONFIG_SECTION, key)
    except Exception:
        return default


def _resolve_device():
    """GPU if available else CPU. Honors DEVICE env / config.ini (cpu/cuda/auto)."""
    dev = os.environ.get("DEVICE", _cfg(Constants.DEVICE_KEY, "auto")).strip().lower()
    if dev == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return dev


DET_SIZE       = (int(float(_cfg(Constants.DET_SIZE_W_KEY, 640))), int(float(_cfg(Constants.DET_SIZE_H_KEY, 640))))
MIN_FACE_PX    = int(float(_cfg(Constants.MIN_FACE_PX_KEY, 28)))
FACE_DET_CONF  = float(_cfg(Constants.FACE_DET_CONF_KEY, 0.40))
BLUR_MIN       = float(_cfg(Constants.BLUR_MIN_KEY, 30.0))
UPSCALE_BELOW  = float(_cfg(Constants.UPSCALE_BELOW_KEY, 220))
MAX_UPSCALE    = float(_cfg(Constants.MAX_UPSCALE_KEY, 4.0))
BEST_FACES_DIR = _cfg(Constants.BEST_FACES_DIR_KEY, "output/best_faces")
REID_THRESHOLD = float(_cfg(Constants.REID_THRESHOLD_KEY, 0.45))

# Composite quality-score weights (should sum to ~1.0, but not enforced).
# No eye-openness term: buffalo_l's 106-point landmark index layout for eye
# contours isn't documented in the package, and guessing wrong indices would
# silently produce a garbage signal -- dropped rather than risk that; weight
# redistributed onto the other four terms.
W_CONF    = float(_cfg(Constants.SCORE_W_CONF_KEY, 0.35))
W_SHARP   = float(_cfg(Constants.SCORE_W_SHARP_KEY, 0.30))
W_SIZE    = float(_cfg(Constants.SCORE_W_SIZE_KEY, 0.15))
W_FRONTAL = float(_cfg(Constants.SCORE_W_FRONTAL_KEY, 0.20))

# Normalization reference points (not hard limits -- values are clamped to [0,1] after scaling).
SHARP_REF_MAX = float(_cfg(Constants.SHARP_REF_MAX_KEY, 400.0))   # Laplacian variance considered "very sharp"
SIZE_REF_MAX  = float(_cfg(Constants.SIZE_REF_MAX_KEY, 200.0))    # face min(w,h) in px considered "large"

GENDER_LABELS = {0: "female", 1: "male"}


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _is_sharp_score(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return var, _clamp01(var / SHARP_REF_MAX)


def _frontal_score(pose) -> float:
    """`pose` is buffalo_l's own [pitch, yaw, roll] in degrees, computed by
    its landmark_3d_68 model (a 3D-to-3D fit against a mean face shape) --
    no separate pose-estimation step needed. Roll (in-plane rotation) is
    ignored: a tilted-but-frontal face is still a usable face for matching."""
    if pose is None:
        return 0.5  # neutral -- neither rewarded nor penalized
    pitch, yaw = float(pose[0]), float(pose[1])
    return _clamp01(1.0 - (abs(pitch) + abs(yaw)) / 90.0)


class FaceExtractor:
    """InsightFace face detection + quality gate + composite scoring, with
    small-crop upscaling for distant people. Uses all 5 buffalo_l sub-models:
    detection, both landmark models (via face.pose), age/gender, recognition."""

    def __init__(self, device: str):
        if FaceAnalysis is None:
            raise ImportError("insightface is not installed. Please install 'insightface' and 'onnxruntime'.")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device != "cpu" \
            else ["CPUExecutionProvider"]
        self.lock = threading.Lock()
        self.face_app = FaceAnalysis(name="buffalo_l", providers=providers)
        self.face_app.prepare(ctx_id=-1 if device == "cpu" else 0, det_size=DET_SIZE)
        logger.info(f"InsightFace buffalo_l loaded (providers: {providers}, det_size: {DET_SIZE})")

    def extract(self, person_crop: np.ndarray):
        """Returns a list of candidates -- bbox (relative to the ORIGINAL,
        non-upscaled person_crop), det_score, face_crop, embedding, age,
        gender, score -- one per face in this crop that passes the gates."""
        if person_crop is None or person_crop.size == 0:
            return []

        h, w = person_crop.shape[:2]
        scale = 1.0
        if min(h, w) < UPSCALE_BELOW:
            scale = min(MAX_UPSCALE, UPSCALE_BELOW / max(1, min(h, w)))

        detect_crop = person_crop
        if scale > 1.0:
            detect_crop = cv2.resize(person_crop, (int(round(w * scale)), int(round(h * scale))),
                                      interpolation=cv2.INTER_CUBIC)

        with self.lock:
            try:
                faces = self.face_app.get(detect_crop)
            except Exception as e:
                logger.warning(f"Face detection failed on a crop ({e}); skipping.")
                return []
        if not faces:
            return []

        results = []
        for face in faces:
            if face.det_score < FACE_DET_CONF:
                continue
            if face.kps is None or len(face.kps) < 3:
                continue

            dfx1, dfy1, dfx2, dfy2 = map(int, face.bbox)
            dfw, dfh = dfx2 - dfx1, dfy2 - dfy1
            if min(dfw, dfh) < MIN_FACE_PX:
                continue

            face_crop = detect_crop[max(0, dfy1):dfy2, max(0, dfx1):dfx2]
            if face_crop.size == 0:
                continue
            sharpness_raw, sharp_score = _is_sharp_score(face_crop)
            if sharpness_raw < BLUR_MIN:
                continue

            frontal_score = _frontal_score(getattr(face, "pose", None))
            size_score = _clamp01(min(dfw, dfh) / SIZE_REF_MAX)
            conf_score = _clamp01(float(face.det_score))

            score = (W_CONF * conf_score + W_SHARP * sharp_score
                     + W_SIZE * size_score + W_FRONTAL * frontal_score)

            fx1, fy1, fx2, fy2 = (int(round(v / scale)) for v in (dfx1, dfy1, dfx2, dfy2))
            gender = getattr(face, "gender", None)
            # kps is in detect_crop's (possibly upscaled) coordinates -- scale
            # back to the same coordinate space as bbox, for consistency.
            landmarks = (np.asarray(face.kps) / scale).round(1).tolist() if face.kps is not None else None
            results.append({
                "bbox": (fx1, fy1, fx2, fy2),
                "det_score": float(face.det_score),
                "face_crop": face_crop,
                "embedding": face.normed_embedding,
                "age": int(face.age) if getattr(face, "age", None) is not None else None,
                "gender": GENDER_LABELS.get(int(gender)) if gender is not None else None,
                "score": score,
                "face_size": [fx2 - fx1, fy2 - fy1],
                "sharpness": round(sharpness_raw, 2),
                "landmarks": landmarks,
            })
        return results


# ---- shared singleton (stateless inference, one per container) ------------
_face_extractor = None
_model_init_lock = threading.Lock()


def _get_face_extractor():
    global _face_extractor
    if _face_extractor is None:
        with _model_init_lock:
            if _face_extractor is None:
                _face_extractor = FaceExtractor(_resolve_device())
    return _face_extractor


# ---- ReID: track_id -> stable global_person_id, per camera -----------------
# _track_to_global[camera_id][track_id] = global_person_id
# _gallery[camera_id][global_person_id] = {"embedding": ..., "score": ...}
_track_to_global = {}
_gallery = {}
_next_global_id = {}
_reid_lock = threading.Lock()


def _cosine_sim(a, b) -> float:
    return float(np.dot(a, b))  # both are already L2-normalized (normed_embedding)


def _resolve_global_id(camera_id, track_id, embedding) -> int:
    """Maps a (possibly unstable) track_id to a stable global_person_id,
    merging it into an existing identity if its face embedding matches one
    already seen on this camera (cosine similarity >= REID_THRESHOLD) --
    this is what recovers identity across a tracker ID reset caused by
    occlusion or a person briefly leaving frame."""
    with _reid_lock:
        track_map = _track_to_global.setdefault(camera_id, {})
        if track_id is not None and track_id in track_map:
            return track_map[track_id]

        gallery = _gallery.setdefault(camera_id, {})
        best_id, best_sim = None, 0.0
        for gid, entry in gallery.items():
            sim = _cosine_sim(embedding, entry["embedding"])
            if sim > best_sim:
                best_sim, best_id = sim, gid

        if best_id is not None and best_sim >= REID_THRESHOLD:
            global_id = best_id
        else:
            _next_global_id.setdefault(camera_id, 0)
            global_id = _next_global_id[camera_id]
            _next_global_id[camera_id] += 1
            gallery[global_id] = {"embedding": embedding, "score": -1.0}

        if track_id is not None:
            track_map[track_id] = global_id
        return global_id


def _update_best_face(camera_id, global_id, candidate) -> bool:
    """Saves `candidate` as the new best face for this global_person_id if
    it beats the current best (or there is none yet). Returns True if files
    were written."""
    os.makedirs(BEST_FACES_DIR, exist_ok=True)
    gallery = _gallery.setdefault(camera_id, {})

    with _reid_lock:
        entry = gallery.get(global_id)
        write = entry is None or candidate["score"] > entry["score"]
        if write:
            gallery[global_id] = {"embedding": candidate["embedding"], "score": candidate["score"]}

    if not write:
        return False

    # camera_id is part of the filename because global_person_id is only
    # unique WITHIN a camera's own gallery (see _resolve_global_id) -- two
    # different cameras can both produce global_person_id 0, and without
    # this they'd silently overwrite each other's best-face file.
    person_tag = f"person_{camera_id}_{global_id}"
    cv2.imwrite(os.path.join(BEST_FACES_DIR, f"{person_tag}.jpg"), candidate["face_crop"])
    np.save(os.path.join(BEST_FACES_DIR, f"{person_tag}.npy"), candidate["embedding"])
    with open(os.path.join(BEST_FACES_DIR, f"{person_tag}.json"), "w") as f:
        json.dump({
            "global_person_id": global_id,
            "camera_id": camera_id,
            "det_score": candidate["det_score"],
            "score": round(candidate["score"], 4),
            "age": candidate["age"],
            "gender": candidate["gender"],
            "face_size": candidate["face_size"],
            "sharpness": candidate["sharpness"],
            "landmarks": candidate["landmarks"],
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, f, indent=2)
    return True


def run_logic(batch: dict) -> dict:
    """Mandatory entrypoint. The BL orchestrator loads this module and calls
    this function by name; do not rename it.

    Expected batch shape (mirrors what AI_VMS's BL orchestrator passes in):
        batch["camera_metadata"]["camera_id"]  -> str
        batch["frame_metadata"]["frame"]       -> raw encoded image bytes OR a decoded np.ndarray
        batch["inference_results"]             -> [[{"xyxy":[...], "class_name":"person",
                                                       "track_id": int|None, ...}, ...]]

    On return, batch["bl_results"] is set to:
        {"faces": [{"global_person_id", "track_id", "bbox", "det_score", "score"}, ...],
         "alert": bool}
    """
    face_extractor = _get_face_extractor()

    camera_id = batch.get("camera_metadata", {}).get("camera_id", "default")
    frame_metadata = batch.get("frame_metadata", {})

    frame = frame_metadata.get("frame")
    if isinstance(frame, (bytes, bytearray)):
        frame = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        batch["bl_results"] = {"faces": [], "alert": False}
        return batch

    detections_per_frame = batch.get("inference_results") or [[]]
    frame_detections = detections_per_frame[0]

    people = []
    for det in frame_detections:
        xyxy = det.get("xyxy")
        if str(det.get("class_name", "")).lower() != "person" or not xyxy or len(xyxy) != 4:
            continue
        people.append((tuple(map(int, xyxy)), det.get("track_id")))

    # All boxes/labels are drawn on a SEPARATE copy (`annotated`), never on
    # `frame` itself -- person/face crops are always cut from the clean,
    # unmarked `frame` so a red/green box border can never bleed into a
    # saved face crop (this was a real bug: cropping from the already-drawn
    # frame let the red person-box border show up at the edge of extracted
    # faces near that boundary).
    annotated = frame.copy()

    for (x1, y1, x2, y2), track_id in people:
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)  # red = person

    faces_out = []
    for (x1, y1, x2, y2), track_id in people:
        person_crop = frame[max(0, y1):y2, max(0, x1):x2]
        for candidate in face_extractor.extract(person_crop):
            global_id = _resolve_global_id(camera_id, track_id, candidate["embedding"])

            fx1, fy1, fx2, fy2 = candidate["bbox"]
            gx1, gy1, gx2, gy2 = x1 + fx1, y1 + fy1, x1 + fx2, y1 + fy2
            cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), (0, 200, 0), 2)  # green = face
            cv2.putText(annotated, f"ID {global_id}", (gx1, max(0, gy1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)

            _update_best_face(camera_id, global_id, candidate)

            faces_out.append({
                "global_person_id": global_id,
                "track_id": track_id,
                "bbox": [gx1, gy1, gx2, gy2],
                "det_score": candidate["det_score"],
                "score": round(candidate["score"], 4),
            })

    # Re-encode back to bytes -- messages travel as encoded bytes over
    # RabbitMQ/msgpack in real AI_VMS, same as every other BL usecase.
    _, encoded = cv2.imencode(".jpg", annotated)
    frame_metadata["frame"] = encoded.tobytes()

    batch["bl_results"] = {
        "faces": faces_out,
        "alert": len(faces_out) > 0,
    }
    return batch
