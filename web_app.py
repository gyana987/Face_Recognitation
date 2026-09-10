"""Web frontend for the missing-person search: upload a reference photo,
configure camera sources, watch live annotated video in the browser, and
confirm/reject candidate matches with buttons instead of a terminal prompt.

Reuses the same detection/matching building blocks as search_multi_camera.py
(imported directly from it) -- this file only replaces the OpenCV window +
terminal y/n with a Flask backend (MJPEG streaming) + HTML/JS frontend.

Run:
    /home/gyana/jupyter_env/bin/python3 web_app.py
    then open http://<this-machine-ip>:5000 in a browser
"""
import logging
import os
import threading
import time
import uuid

from search_multi_camera import (
    BASE_DIR, PINK_BGR, FaceMatcher, cosine_sim, crop_upscaled,
    load_person_model, resolve_device,
)

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, Response

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("WebApp")

REFERENCE_DIR = os.path.join(BASE_DIR, "input", "reference")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "search_results")
os.makedirs(REFERENCE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DET_SIZE = (640, 640)
MIN_FACE_PX = 28
UPSCALE_BELOW = 220
MAX_UPSCALE = 4.0
PERSON_CONF = 0.25
PERSON_IMGSZ = 960

# ---------------------------------------------------------------------------
# Global state: models are loaded once at startup and reused across every
# start/stop cycle; only the camera threads themselves are (re)created per
# search session.
# ---------------------------------------------------------------------------
models_lock = threading.Lock()
models = {"device": None, "person_model": None, "matcher": None, "ready": False}

state_lock = threading.Lock()
frames_lock = threading.Lock()

state = {
    "running": False,
    "ref_embedding": None,
    "ref_image_path": None,
    "ref_score": None,
    "camera_ids": [],
    "similarity_threshold": 0.45,
    "every": 5,
    "alert": {"active": False, "text": "IDLE"},
    "pending_candidate": None,   # {"id", "camera_id", "similarity", "frame_idx"}
    "cameras_status": {},        # camera_id -> {"candidates_shown", "confirmed", "error"}
    "log": [],                   # rolling list of recent event strings
    "stop_event": None,
    "prompt_lock": None,
    "decision_event": None,
    "decision": None,
}
latest_frames = {}  # camera_id -> np.ndarray (annotated BGR frame)


def log_event(msg):
    logger.info(msg)
    with state_lock:
        state["log"].append(msg)
        state["log"] = state["log"][-50:]


def load_models_background():
    device = resolve_device("auto")
    logger.info(f"device: {device}")
    logger.info("loading person detector...")
    person_model = load_person_model(device)
    logger.info("loading InsightFace buffalo_l...")
    matcher = FaceMatcher(device, DET_SIZE, MIN_FACE_PX)
    with models_lock:
        models["device"] = device
        models["person_model"] = person_model
        models["matcher"] = matcher
        models["ready"] = True
    log_event(f"Models loaded (device={device}). Ready to search.")


threading.Thread(target=load_models_background, daemon=True).start()


# ---------------------------------------------------------------------------
# Per-camera worker -- same detection/matching logic as
# search_multi_camera.run_one_camera_search, but the "confirm" step blocks
# on a threading.Event set by the /api/confirm route instead of input().
# ---------------------------------------------------------------------------
def web_prompt_confirm(camera_id, similarity, frame_idx, annotated):
    candidate_id = uuid.uuid4().hex[:8]
    out_path = os.path.join(OUTPUT_DIR, f"candidate_{camera_id}_{candidate_id}_sim{similarity:.2f}.jpg")
    cv2.imwrite(out_path, annotated)

    decision_event = threading.Event()
    with state_lock:
        state["pending_candidate"] = {
            "id": candidate_id, "camera_id": camera_id, "similarity": round(similarity, 3),
            "frame_idx": frame_idx, "image_url": f"/candidate_image/{os.path.basename(out_path)}",
        }
        state["decision_event"] = decision_event
        state["decision"] = None
    log_event(f"[{camera_id}] candidate found (similarity={similarity:.3f}) -- waiting for confirm/reject")

    decision_event.wait()  # blocks until /api/confirm sets it

    with state_lock:
        decision = state["decision"]
        state["pending_candidate"] = None
    return decision == "yes"


def camera_worker(camera_id, input_video, start_time_sec, person_model, matcher, ref_embedding,
                   stop_event, prompt_lock, similarity_threshold, every):
    log = logging.getLogger(f"Camera[{camera_id}]")
    # input_video is either a webcam's integer device index (e.g. "0") or
    # an .mp4 file path. A camera id is live (no pacing needed); an .mp4
    # file gets real-time pacing so it doesn't race through faster than it
    # was actually recorded.
    is_camera_id = str(input_video).strip().isdigit()
    is_file_source = not is_camera_id and str(input_video).strip().lower().endswith(".mp4")
    source = int(input_video) if is_camera_id else input_video

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log.error(f"Could not open video source: {input_video}")
        with state_lock:
            state["cameras_status"][camera_id] = {"candidates_shown": 0, "confirmed": False,
                                                    "error": f"Could not open: {input_video}"}
        log_event(f"[{camera_id}] ERROR: could not open {input_video}")
        return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if start_time_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time_sec * 1000)

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    pace_seconds_per_kept_frame = every / video_fps if is_file_source else 0.0
    next_frame_due_at = time.monotonic()

    rejected_track_ids = set()
    frame_idx = 0
    candidates_shown = 0
    confirmed = False
    log_event(f"[{camera_id}] started -- {input_video}")

    while not stop_event.is_set():
        if not cap.grab():
            break
        frame_idx += 1
        if frame_idx % every != 0:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break

        if pace_seconds_per_kept_frame > 0:
            next_frame_due_at += pace_seconds_per_kept_frame
            delay = next_frame_due_at - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_frame_due_at = time.monotonic()

        detections = person_model.run_inference(
            [frame],
            {"tracking": True, "confidence": PERSON_CONF, "imgsz": PERSON_IMGSZ, "tracker_key": camera_id},
        )[0]

        annotated = frame.copy()
        match_this_frame = None

        for det in detections:
            if str(det.get("class_name", "")).lower() != "person":
                continue
            track_id = det.get("track_id")
            if track_id is not None and track_id in rejected_track_ids:
                continue

            x1, y1, x2, y2 = map(int, det["xyxy"])
            person_crop = frame[max(0, y1):y2, max(0, x1):x2]
            if person_crop.size == 0:
                continue

            detect_crop, scale = crop_upscaled(person_crop, UPSCALE_BELOW, MAX_UPSCALE)
            embedding, face_bbox, det_score = matcher.best_face_embedding(detect_crop)
            if embedding is None or face_bbox is None:
                continue

            fx1, fy1, fx2, fy2 = face_bbox
            gx1, gy1, gx2, gy2 = (x1 + int(round(fx1 / scale)), y1 + int(round(fy1 / scale)),
                                   x1 + int(round(fx2 / scale)), y1 + int(round(fy2 / scale)))

            similarity = cosine_sim(embedding, ref_embedding)
            is_match = similarity >= similarity_threshold

            if is_match:
                match_this_frame = (track_id, similarity)
                cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), PINK_BGR, 3)
                cv2.putText(annotated, f"MATCH sim={similarity:.2f}", (gx1, max(0, gy1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, PINK_BGR, 2, cv2.LINE_AA)
            else:
                cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), (0, 200, 0), 2)

        with frames_lock:
            latest_frames[camera_id] = annotated

        if match_this_frame is not None:
            track_id, similarity = match_this_frame
            with state_lock:
                state["alert"] = {"active": True, "text": f"PERSON FOUND on {camera_id} (sim={similarity:.2f})"}
            with prompt_lock:
                if stop_event.is_set():
                    break
                candidates_shown += 1
                if web_prompt_confirm(camera_id, similarity, frame_idx, annotated):
                    confirmed = True
                    stop_event.set()
                    confirmed_path = os.path.join(OUTPUT_DIR, f"MATCH_CONFIRMED_{camera_id}.jpg")
                    cv2.imwrite(confirmed_path, annotated)
                    log_event(f"[{camera_id}] MATCH CONFIRMED (similarity={similarity:.3f}). Stopping all cameras.")
                    with state_lock:
                        state["alert"] = {"active": True, "text": f"MATCH CONFIRMED on {camera_id}"}
                else:
                    if track_id is not None:
                        rejected_track_ids.add(track_id)
                    log_event(f"[{camera_id}] candidate rejected, continuing search")
                    with state_lock:
                        state["alert"] = {"active": False, "text": "SEARCHING..."}

        if stop_event.is_set():
            break

    cap.release()
    with state_lock:
        state["cameras_status"][camera_id] = {"candidates_shown": candidates_shown, "confirmed": confirmed}
    log_event(f"[{camera_id}] stopped -- {candidates_shown} candidate(s), "
              + ("CONFIRMED" if confirmed else "no confirmed match"))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/ready")
def api_ready():
    with models_lock:
        return jsonify({"ready": models["ready"], "device": models["device"]})


@app.route("/api/upload_reference", methods=["POST"])
def api_upload_reference():
    with models_lock:
        if not models["ready"]:
            return jsonify({"ok": False, "error": "Models still loading, try again in a few seconds."}), 503
        matcher = models["matcher"]

    file = request.files.get("image")
    if not file or file.filename == "":
        return jsonify({"ok": False, "error": "No file uploaded."}), 400

    ext = os.path.splitext(file.filename)[1] or ".jpg"
    save_path = os.path.join(REFERENCE_DIR, f"upload_{uuid.uuid4().hex[:8]}{ext}")
    file.save(save_path)

    img = cv2.imread(save_path)
    if img is None:
        return jsonify({"ok": False, "error": "Could not read the uploaded file as an image."}), 400

    upscaled, _ = crop_upscaled(img, UPSCALE_BELOW, MAX_UPSCALE)
    embedding, _, score = matcher.best_face_embedding(upscaled)
    if embedding is None:
        return jsonify({"ok": False, "error": "No usable face found in that photo. Try a clearer photo "
                                               "(a full-person crop with some body context works better "
                                               "than a tight face-only crop)."}), 400

    with state_lock:
        state["ref_embedding"] = embedding
        state["ref_image_path"] = save_path
        state["ref_score"] = round(float(score), 3)
    log_event(f"Reference photo set (face det_score={score:.3f}).")
    return jsonify({"ok": True, "det_score": round(float(score), 3), "image_url": f"/reference_image"})


@app.route("/reference_image")
def reference_image():
    with state_lock:
        path = state["ref_image_path"]
    if not path or not os.path.exists(path):
        return "", 404
    with open(path, "rb") as f:
        data = f.read()
    return Response(data, mimetype="image/jpeg")


@app.route("/candidate_image/<name>")
def candidate_image(name):
    path = os.path.join(OUTPUT_DIR, name)
    if not os.path.exists(path):
        return "", 404
    with open(path, "rb") as f:
        data = f.read()
    return Response(data, mimetype="image/jpeg")


@app.route("/api/start", methods=["POST"])
def api_start():
    with models_lock:
        if not models["ready"]:
            return jsonify({"ok": False, "error": "Models still loading, try again in a few seconds."}), 503
        person_model, matcher = models["person_model"], models["matcher"]

    with state_lock:
        if state["running"]:
            return jsonify({"ok": False, "error": "A search is already running. Stop it first."}), 400
        ref_embedding = state["ref_embedding"]
    if ref_embedding is None:
        return jsonify({"ok": False, "error": "Upload a missing-person reference photo first."}), 400

    payload = request.get_json(force=True)
    cameras = payload.get("cameras", [])
    if not cameras:
        return jsonify({"ok": False, "error": "Add at least one camera."}), 400
    similarity_threshold = float(payload.get("similarity_threshold", 0.45))
    every = max(1, int(payload.get("every", 5)))

    stop_event = threading.Event()
    prompt_lock = threading.Lock()
    with state_lock:
        state["running"] = True
        state["stop_event"] = stop_event
        state["prompt_lock"] = prompt_lock
        state["similarity_threshold"] = similarity_threshold
        state["every"] = every
        state["camera_ids"] = [c["camera_id"] for c in cameras]
        state["cameras_status"] = {}
        state["pending_candidate"] = None
        state["alert"] = {"active": False, "text": "SEARCHING..."}
    with frames_lock:
        latest_frames.clear()

    for cam in cameras:
        t = threading.Thread(
            target=camera_worker,
            args=(cam["camera_id"], cam["input_video"], float(cam.get("start_time_sec", 0) or 0),
                  person_model, matcher, ref_embedding, stop_event, prompt_lock,
                  similarity_threshold, every),
            name=f"web-search-{cam['camera_id']}", daemon=True,
        )
        t.start()

    log_event(f"Search started on cameras: {', '.join(state['camera_ids'])}")

    def watch_finish():
        # Flips running back to False once every camera thread has stopped
        # (e.g. all videos ended, or a match was confirmed).
        while True:
            time.sleep(1)
            with state_lock:
                if len(state["cameras_status"]) >= len(state["camera_ids"]):
                    state["running"] = False
                    return

    threading.Thread(target=watch_finish, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/confirm", methods=["POST"])
def api_confirm():
    payload = request.get_json(force=True)
    decision = payload.get("decision")
    if decision not in ("yes", "no"):
        return jsonify({"ok": False, "error": "decision must be 'yes' or 'no'"}), 400
    with state_lock:
        if state["decision_event"] is None or state["pending_candidate"] is None:
            return jsonify({"ok": False, "error": "No pending candidate to confirm."}), 400
        state["decision"] = decision
        state["decision_event"].set()
    return jsonify({"ok": True})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    with state_lock:
        stop_event = state["stop_event"]
        decision_event = state["decision_event"]
        if state["pending_candidate"] is not None:
            state["decision"] = "no"  # unblock anyone waiting on a prompt
    if decision_event is not None:
        decision_event.set()
    if stop_event is not None:
        stop_event.set()
    log_event("Stop requested.")
    return jsonify({"ok": True})


@app.route("/api/status")
def api_status():
    with state_lock:
        return jsonify({
            "running": state["running"],
            "alert": state["alert"],
            "pending_candidate": state["pending_candidate"],
            "cameras_status": state["cameras_status"],
            "camera_ids": state["camera_ids"],
            "ref_score": state["ref_score"],
            "log": state["log"][-20:],
        })


_BLANK_FRAME = np.full((480, 640, 3), 40, dtype=np.uint8)


def gen_frames(camera_id):
    while True:
        with frames_lock:
            frame = latest_frames.get(camera_id, _BLANK_FRAME)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(0.05)


@app.route("/video/<camera_id>")
def video_feed(camera_id):
    return Response(gen_frames(camera_id), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
