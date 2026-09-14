"""Search MULTIPLE camera feeds for one missing person at once, each with
its own live window, sharing one loaded face-recognition model.

Config comes entirely from config/search_cameras.ini: one missing-person
photo, a list of camera_ids, and an [camera:<id>] section per camera giving
its video path. Add/remove cameras by editing that file, not this script.

Threading design:
    - Each camera gets its OWN person detector + tracker instance (passing
      its camera_id as tracker_key to run_inference) -- ByteTrack's
      persist=True state lives inside the model object, so two cameras
      sharing one tracker would corrupt both cameras' track_ids.
    - ONE shared FaceMatcher (InsightFace buffalo_l) across all cameras --
      it has no state between calls and is internally lock-protected, so
      sharing it just means concurrent cameras queue for GPU time instead
      of doubling GPU memory use.
    - Terminal confirmation (the y/n prompt) is serialized with a shared
      lock: if two cameras find a candidate at the same moment, only one
      prompt is active on stdin at a time. The OTHER camera's live window
      simply pauses on its current frame until it's its turn -- there's
      only one person watching, after all.
    - The moment ANY camera's candidate is confirmed with "y", a shared
      stop_event tells every camera's loop to stop, since the point of the
      search (finding the person) is done.

Usage:
    python search_multi_camera.py
    python search_multi_camera.py --config config/my_search_cameras.ini
"""
import argparse
import configparser
import importlib.util
import logging
import math
import os
import sys
import threading
import time


def _ensure_gpu_libs_on_path():
    """Local-machine cuDNN/LD_LIBRARY_PATH fix -- must run before any
    torch/ultralytics/onnxruntime import."""
    if os.environ.get("_PFP_GPU_LIBS_SET") == "1":
        return
    lib_dirs = []
    for pkg in ("nvidia.cu13", "nvidia.cudnn"):
        spec = importlib.util.find_spec(pkg)
        if spec and spec.submodule_search_locations:
            lib_dirs.append(os.path.join(list(spec.submodule_search_locations)[0], "lib"))
    if not lib_dirs:
        return
    os.environ["LD_LIBRARY_PATH"] = ":".join(lib_dirs + [os.environ.get("LD_LIBRARY_PATH", "")])
    os.environ["_PFP_GPU_LIBS_SET"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


_ensure_gpu_libs_on_path()

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PINK_BGR = (203, 192, 255)  # OpenCV is BGR; this is "pink" (255,192,203 in RGB)
GRID_WINDOW = "Missing-Person Search -- Live (all cameras)"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("MultiCameraSearch")


def resolve_device(device_cfg: str) -> str:
    if device_cfg != "auto":
        return device_cfg
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def load_person_model(device: str):
    """Loads the same person detector the main pipeline uses, via the same
    dynamic-load mechanism run_local.py uses."""
    stage_path = os.path.join(BASE_DIR, "AI_Orchestration")
    sys.path.insert(0, stage_path)
    try:
        spec = importlib.util.spec_from_file_location(
            "yolo_inference", os.path.join(BASE_DIR, "AI_Orchestration", "Detection", "YOLO", "inference.py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["yolo_inference"] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(stage_path)


def crop_upscaled(img, upscale_below, max_upscale):
    """Upsamples a small crop before face detection (recovers detectable
    detail from a distant/tiny person). Returns (image, scale factor applied)."""
    h, w = img.shape[:2]
    scale = 1.0
    if min(h, w) < upscale_below:
        scale = min(max_upscale, upscale_below / max(1, min(h, w)))
    if scale > 1.0:
        return cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_CUBIC), scale
    return img, scale


class FaceMatcher:
    """Thin InsightFace buffalo_l wrapper: detect the best face in a crop and
    return its embedding, with a quality gate (min size). Thread-safe
    (guarded by an internal lock) so ONE instance can be safely shared
    across multiple camera threads -- InsightFace has no state between
    calls, so sharing just means concurrent cameras queue for GPU time
    rather than needing a separate loaded model per camera."""

    def __init__(self, device: str, det_size, min_face_px: int):
        from insightface.app import FaceAnalysis
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device != "cpu" \
            else ["CPUExecutionProvider"]
        # allowed_modules restricts buffalo_l to only the 2 models we actually
        # use (detection for the bbox, recognition for the embedding) --
        # without this, InsightFace also runs landmark_2d_106, landmark_3d_68
        # and genderage per face on every frame even though nothing reads
        # their output, which matters a lot when face-matching falls back to
        # CPU (see FaceMatcher class docstring).
        self.face_app = FaceAnalysis(name="buffalo_l", providers=providers,
                                      allowed_modules=["detection", "recognition"])
        self.face_app.prepare(ctx_id=-1 if device == "cpu" else 0, det_size=det_size)
        self.min_face_px = min_face_px
        self.lock = threading.Lock()

    def best_face_embedding(self, img):
        """Returns (embedding, bbox_in_img_coords, det_score) for the best
        face in `img`, or (None, None, None) if none passes the gate."""
        with self.lock:
            faces = self.face_app.get(img)
        if not faces:
            return None, None, None
        face = max(faces, key=lambda f: f.det_score)
        x1, y1, x2, y2 = map(int, face.bbox)
        if min(x2 - x1, y2 - y1) < self.min_face_px:
            return None, None, None
        return face.normed_embedding, (x1, y1, x2, y2), float(face.det_score)


def cosine_sim(a, b) -> float:
    return float(np.dot(a, b))  # both L2-normalized already


def prompt_yes_no(prompt: str) -> bool:
    while True:
        answer = input(prompt).strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("Please answer 'y' or 'n'.")


def load_cameras_config(path):
    parser = configparser.ConfigParser()
    if not parser.read(path):
        raise SystemExit(f"Config file not found: {path}")
    g = parser["general"]
    camera_ids = [c.strip() for c in g.get("CAMERA_IDS", "").split(",") if c.strip()]
    if not camera_ids:
        raise SystemExit(f"No cameras listed in CAMERA_IDS in {path}")

    cameras = []
    for cam_id in camera_ids:
        section = f"camera:{cam_id}"
        if section not in parser:
            raise SystemExit(f"Missing [{section}] section in {path} for camera '{cam_id}' listed in CAMERA_IDS")
        cameras.append({
            "camera_id": cam_id,
            "input_video": parser[section].get("INPUT_VIDEO"),
            "start_time_sec": parser[section].getfloat("START_TIME_SEC", fallback=0.0),
        })

    return {
        "cameras": cameras,
        "missing_person_image": g.get("MISSING_PERSON_IMAGE"),
        "output_dir": g.get("OUTPUT_DIR", "output/search_results"),
        "similarity_threshold": g.getfloat("SIMILARITY_THRESHOLD", fallback=0.45),
        "every": g.getint("EVERY", fallback=1),
        "device": g.get("DEVICE", fallback="auto"),
        "person_conf": g.getfloat("PERSON_CONF", fallback=0.25),
        "person_imgsz": g.getint("PERSON_IMGSZ", fallback=1280),
        "det_size": (g.getint("DET_SIZE_W", fallback=640), g.getint("DET_SIZE_H", fallback=640)),
        "min_face_px": g.getint("MIN_FACE_PX", fallback=28),
        "upscale_below": g.getfloat("UPSCALE_BELOW", fallback=220),
        "max_upscale": g.getfloat("MAX_UPSCALE", fallback=4.0),
        "save_window_video": g.getboolean("SAVE_WINDOW_VIDEO", fallback=True),
        "window_video_fps": g.getfloat("WINDOW_VIDEO_FPS", fallback=8.0),
    }


def run_one_camera_search(camera_cfg, general_cfg, person_model, matcher, ref_embedding,
                           stop_event, prompt_lock, results, results_lock,
                           latest_frames, frames_lock, alert_state, alert_lock):
    camera_id = camera_cfg["camera_id"]
    video_path = camera_cfg["input_video"]
    log = logging.getLogger(f"Camera[{camera_id}]")

    # INPUT_VIDEO is either a webcam's integer device index (e.g. "0") or
    # an .mp4 file path. A camera id is live (no pacing needed -- frames
    # already arrive in real time); an .mp4 file gets real-time pacing so
    # it doesn't race through faster than it was actually recorded.
    is_camera_id = str(video_path).strip().isdigit()
    is_file_source = not is_camera_id and str(video_path).strip().lower().endswith(".mp4")
    source = int(video_path) if is_camera_id else video_path
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log.error(f"Could not open video source: {video_path}")
        return
    # For a live stream, OpenCV buffers frames internally -- without this,
    # a detector slower than the camera's frame rate falls further and
    # further behind real time as the buffer fills. Keeping only 1 frame
    # buffered means we always grab the newest frame, staying live.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    start_time_sec = camera_cfg.get("start_time_sec", 0.0)
    if start_time_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time_sec * 1000)

    # A live camera/RTSP feed is already paced by the hardware -- frames
    # simply arrive at real-world speed. A VIDEO FILE has no such pacing:
    # if the GPU processes it faster than the video's own frame rate, it
    # races through and looks sped-up instead of "live". Throttle file
    # sources back down to the source video's real playback speed.
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    pace_seconds_per_kept_frame = general_cfg["every"] / video_fps if is_file_source else 0.0
    next_frame_due_at = time.monotonic()

    rejected_track_ids = set()
    frame_idx = 0
    candidates_shown = 0
    confirmed = False

    log.info(f"Started -- {video_path}" + (f" (seeking to {start_time_sec:.0f}s)" if start_time_sec > 0 else ""))

    while not stop_event.is_set():
        # grab() just advances the stream without fully decoding the frame --
        # much cheaper than read() for the frames we're about to skip anyway.
        # Only retrieve() (full decode) the one frame in every EVERY we keep.
        if not cap.grab():
            break
        frame_idx += 1
        if frame_idx % general_cfg["every"] != 0:
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
                # Fell behind real time (e.g. a slow detection, or the y/n
                # prompt paused us) -- don't try to "catch up" by racing
                # through frames; just resume pacing from now.
                next_frame_due_at = time.monotonic()

        # tracker_key=camera_id keeps this camera's tracking state fully
        # separate from every other camera's (see module docstring).
        detections = person_model.run_inference(
            [frame],
            {"tracking": True, "confidence": general_cfg["person_conf"],
             "imgsz": general_cfg["person_imgsz"], "tracker_key": camera_id},
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

            detect_crop, scale = crop_upscaled(person_crop, general_cfg["upscale_below"],
                                                   general_cfg["max_upscale"])
            embedding, face_bbox, det_score = matcher.best_face_embedding(detect_crop)
            if embedding is None or face_bbox is None:
                continue

            fx1, fy1, fx2, fy2 = face_bbox
            gx1, gy1, gx2, gy2 = (x1 + int(round(fx1 / scale)), y1 + int(round(fy1 / scale)),
                                   x1 + int(round(fx2 / scale)), y1 + int(round(fy2 / scale)))

            similarity = cosine_sim(embedding, ref_embedding)
            is_match = similarity >= general_cfg["similarity_threshold"]

            if is_match:
                match_this_frame = (gx1, gy1, gx2, gy2, track_id, similarity)
                cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), PINK_BGR, 3)
                cv2.putText(annotated, f"MATCH sim={similarity:.2f}", (gx1, max(0, gy1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, PINK_BGR, 2, cv2.LINE_AA)
            else:
                cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), (0, 200, 0), 2)

        with frames_lock:
            latest_frames[camera_id] = annotated

        if match_this_frame is not None:
            gx1, gy1, gx2, gy2, track_id, similarity = match_this_frame
            with alert_lock:
                alert_state["active"] = True
                alert_state["text"] = f"PERSON FOUND\ncamera: {camera_id}\nsim={similarity:.2f}"
            with prompt_lock:  # only one camera's operator prompt active at a time
                if stop_event.is_set():
                    break  # another camera already confirmed while we waited for the lock
                candidates_shown += 1
                out_path = os.path.join(
                    general_cfg["output_dir"],
                    f"candidate_{camera_id}_{candidates_shown:03d}_frame{frame_idx}_sim{similarity:.2f}.jpg",
                )
                os.makedirs(general_cfg["output_dir"], exist_ok=True)
                cv2.imwrite(out_path, annotated)
                print(f"\n[{camera_id}] candidate {candidates_shown}: frame {frame_idx}, "
                      f"similarity={similarity:.3f} -> saved {out_path}")

                if prompt_yes_no(f"[{camera_id}] Is this the missing person? [y/n]: "):
                    confirmed = True
                    stop_event.set()
                    confirmed_path = os.path.join(general_cfg["output_dir"], f"MATCH_CONFIRMED_{camera_id}.jpg")
                    cv2.imwrite(confirmed_path, annotated)
                    print(f"\nMatch confirmed on {camera_id} at frame {frame_idx}, "
                          f"similarity={similarity:.3f}. Saved {confirmed_path}. Stopping all cameras.")
                    with alert_lock:
                        alert_state["text"] = f"MATCH CONFIRMED\ncamera: {camera_id}\nsim={similarity:.2f}"
                else:
                    if track_id is not None:
                        rejected_track_ids.add(track_id)
                    print(f"[{camera_id}] Continuing search...")
                    with alert_lock:
                        alert_state["active"] = False
                        alert_state["text"] = "SEARCHING..."

        if stop_event.is_set():
            break

    cap.release()
    log.info(f"Stopped -- {candidates_shown} candidate(s) shown"
             + (", CONFIRMED match" if confirmed else ", no confirmed match"))

    with results_lock:
        results[camera_id] = {"candidates_shown": candidates_shown, "confirmed": confirmed}


def _make_tile(frame, camera_id, tile_w, tile_h):
    """Letterbox `frame` (or a black placeholder) into a tile_w x tile_h tile,
    keeping aspect ratio, with the camera id burned into the corner."""
    tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    if frame is None:
        cv2.putText(tile, f"{camera_id}: waiting for frame...", (16, tile_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 160, 160), 2, cv2.LINE_AA)
        return tile

    h, w = frame.shape[:2]
    scale = min(tile_w / w, tile_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(frame, (new_w, new_h))
    x_off, y_off = (tile_w - new_w) // 2, (tile_h - new_h) // 2
    tile[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    cv2.rectangle(tile, (x_off, y_off), (x_off + 150, y_off + 28), (0, 0, 0), -1)
    cv2.putText(tile, camera_id, (x_off + 6, y_off + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return tile


def build_camera_grid(latest_frames, camera_ids, tile_w=800, tile_h=450):
    """Combine every camera's latest annotated frame into ONE grid image
    (2 cams -> side by side, more cams -> extra rows) -- no header, no
    reference photo, just the live camera tiles."""
    cols = math.ceil(math.sqrt(len(camera_ids))) if camera_ids else 1
    rows = math.ceil(len(camera_ids) / cols) if camera_ids else 1

    tiles = [_make_tile(latest_frames.get(cid), cid, tile_w, tile_h) for cid in camera_ids]
    while len(tiles) < rows * cols:
        tiles.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))

    grid_rows = [np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)]
    return np.vstack(grid_rows)


def _make_alert_panel(alert_text, active, w, h):
    """Dedicated alert box for the window's empty right-hand corner: plain
    dark 'SEARCHING...' most of the time, flips to a solid pink 'PERSON
    FOUND' panel the moment any camera gets a candidate -- separate from
    the live video so it doesn't cover the footage being watched."""
    panel = np.full((h, w, 3), (40, 40, 40), dtype=np.uint8)
    color = PINK_BGR if active else (90, 90, 90)
    text_color = (0, 0, 0) if active else (200, 200, 200)
    cv2.rectangle(panel, (0, 0), (w - 1, h - 1), color, -1 if active else 3)

    y = 40
    for line in alert_text.split("\n"):
        cv2.putText(panel, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    text_color, 2, cv2.LINE_AA)
        y += 34
    return panel


def build_display(latest_frames, camera_ids, ref_img, header_text, alert_state,
                   tile_w=800, tile_h=450, bottom_h=220):
    """Full demo frame: cam1/cam2 live grid on top, and below it -- same
    width -- the 'MISSING PERSON' reference photo and the alert box side
    by side, with a plain title header on top (no clock -- it made the
    header look "live" without meaning anything)."""
    camera_grid = build_camera_grid(latest_frames, camera_ids, tile_w=tile_w, tile_h=tile_h)

    total_w = camera_grid.shape[1]
    ref_w = total_w // 2
    alert_w = total_w - ref_w

    ref_tile = _make_tile(ref_img, "MISSING PERSON", ref_w, bottom_h)
    cv2.rectangle(ref_tile, (0, 0), (ref_w - 1, bottom_h - 1), (0, 0, 255), 4)

    alert_panel = _make_alert_panel(alert_state["text"], alert_state["active"], alert_w, bottom_h)

    bottom_row = np.hstack([ref_tile, alert_panel])
    body = np.vstack([camera_grid, bottom_row])

    header_h = 40
    header = np.zeros((header_h, body.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, header_text, (16, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([header, body])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=os.path.join(BASE_DIR, "config", "search_cameras.ini"),
                     help="Path to search_cameras.ini")
    args = ap.parse_args()

    cfg = load_cameras_config(args.config)
    device = resolve_device(cfg["device"])
    logger.info(f"device: {device}")

    logger.info("loading person detector...")
    person_model = load_person_model(device)

    logger.info("loading InsightFace buffalo_l (shared across all cameras)...")
    matcher = FaceMatcher(device, cfg["det_size"], cfg["min_face_px"])

    ref_img_display = cv2.imread(cfg["missing_person_image"])
    if ref_img_display is None:
        raise SystemExit(f"Could not read MISSING_PERSON_IMAGE: {cfg['missing_person_image']}")
    ref_img, _ = crop_upscaled(ref_img_display, cfg["upscale_below"], cfg["max_upscale"])
    ref_embedding, _, ref_score = matcher.best_face_embedding(ref_img)
    if ref_embedding is None:
        raise SystemExit(
            f"No usable face found in {cfg['missing_person_image']}. Use a clearer photo "
            f"(a full person crop with some body context works better than a tight face-only crop)."
        )
    logger.info(f"reference face found (det_score={ref_score:.3f}). "
                f"Searching {len(cfg['cameras'])} camera(s): {', '.join(c['camera_id'] for c in cfg['cameras'])}")

    stop_event = threading.Event()
    prompt_lock = threading.Lock()
    results = {}
    results_lock = threading.Lock()
    latest_frames = {}
    frames_lock = threading.Lock()
    alert_state = {"active": False, "text": "SEARCHING..."}
    alert_lock = threading.Lock()

    threads = []
    for camera_cfg in cfg["cameras"]:
        t = threading.Thread(
            target=run_one_camera_search,
            args=(camera_cfg, cfg, person_model, matcher, ref_embedding,
                  stop_event, prompt_lock, results, results_lock,
                  latest_frames, frames_lock, alert_state, alert_lock),
            name=f"search-{camera_cfg['camera_id']}",
        )
        t.start()
        threads.append(t)

    # Qt/OpenCV's highgui backend is not thread-safe -- all imshow/waitKey
    # calls must happen on a single thread. Worker threads only write their
    # latest annotated frame into latest_frames; this main thread is the
    # only one that ever touches the display, and combines every camera's
    # latest frame into ONE grid so all cameras show in a single window
    # (easier to watch live during a demo than juggling separate windows).
    camera_ids_order = [c["camera_id"] for c in cfg["cameras"]]
    n = max(1, len(camera_ids_order))
    cols = math.ceil(math.sqrt(n))
    tile_w = min(960, 1900 // cols)
    tile_h = int(tile_w * 9 / 16)
    bottom_h = tile_h  # same size as a cam1/cam2 tile, so the reference photo is clearly visible

    header_text = f"MISSING PERSON SEARCH  |  cameras: {', '.join(camera_ids_order)}"
    logger.info(f"Live window: '{GRID_WINDOW}' -- press q in the window to stop.")

    # Record exactly what the window shows -- cameras + reference photo +
    # alert box, all composited -- so the demo can be replayed afterwards
    # without re-running the live search.
    video_writer = None
    video_path = None
    if cfg["save_window_video"]:
        os.makedirs(cfg["output_dir"], exist_ok=True)
        video_path = os.path.join(cfg["output_dir"], "window_output.mp4")
        logger.info(f"Recording window output to {video_path}")

    try:
        while any(t.is_alive() for t in threads):
            with frames_lock:
                snapshot = dict(latest_frames)
            with alert_lock:
                alert_snapshot = dict(alert_state)
            display = build_display(snapshot, camera_ids_order, ref_img_display, header_text,
                                     alert_snapshot, tile_w=tile_w, tile_h=tile_h, bottom_h=bottom_h)
            cv2.imshow(GRID_WINDOW, display)

            if cfg["save_window_video"]:
                if video_writer is None:
                    h, w = display.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(video_path, fourcc, cfg["window_video_fps"], (w, h))
                video_writer.write(display)

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                logger.info("q pressed -- stopping all cameras")
                stop_event.set()
    finally:
        cv2.destroyAllWindows()
        if video_writer is not None:
            video_writer.release()
            logger.info(f"Window output saved to {video_path}")

    for t in threads:
        t.join()

    logger.info("All cameras finished.")
    any_confirmed = any(r["confirmed"] for r in results.values())
    for camera_id, r in results.items():
        status = "CONFIRMED MATCH" if r["confirmed"] else f"{r['candidates_shown']} candidate(s), no match"
        logger.info(f"  {camera_id}: {status}")
    if not any_confirmed:
        logger.info("No camera confirmed a match.")


if __name__ == "__main__":
    main()
