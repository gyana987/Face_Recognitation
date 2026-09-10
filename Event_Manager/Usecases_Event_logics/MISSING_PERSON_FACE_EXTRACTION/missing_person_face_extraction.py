"""MISSING_PERSON_FACE_EXTRACTION event usecase.

Follows the AI_VMS Event_Manager usecase contract: a module-level
run_event_logic(batch, usecase_name) entrypoint. Reads batch["bl_results"]["alert"]
(set by the BL stage's run_logic), de-duplicates repeat alerts per camera with
a cooldown, and saves a short evidence clip built from a rolling per-camera
frame buffer.

In real AI_VMS this stage instead calls:
    services.event_duplication_handler.EventDuplicationHandler().is_duplicate(...)
    services.websocket_client.ws_client.send_event(payload)
    services.db_service.db_service.process_and_save_events(batch, usecase_name, ...)
Those services depend on infra (Postgres, the websocket server) this
standalone/local setup doesn't have, so this module implements the same shape
locally (in-memory cooldown + rolling-buffer evidence clip) -- swap the three
calls above in when porting into AI_VMS, keeping this module's structure/
entrypoint the same. Real AI_VMS's Event orchestrator also already passes a
BATCH of several recent frames per call (unlike this local single-frame-per-call
setup), so its real usecases build evidence clips directly from that batch
instead of keeping their own rolling buffer -- see LOITERING_DETECTION's event
module in AI_VMS for that pattern.
"""

import logging
import os
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

logger = logging.getLogger("MissingPersonFaceExtractionEvent")

ALERT_COOLDOWN_SECONDS = 10.0
CLIP_SECONDS = 10.0
DEFAULT_FPS = 25.0
EVIDENCE_DIR = "output/evidence"

_last_alert_time = {}     # camera_id -> last alert timestamp
_frame_buffers = {}       # camera_id -> deque of recent frames


def _is_duplicate(camera_id: str) -> bool:
    now = time.time()
    last = _last_alert_time.get(camera_id, 0)
    if now - last < ALERT_COOLDOWN_SECONDS:
        return True
    _last_alert_time[camera_id] = now
    return False


def _get_buffer(camera_id: str, fps: float) -> deque:
    buf = _frame_buffers.get(camera_id)
    if buf is None:
        buf = deque(maxlen=max(1, int(round(fps * CLIP_SECONDS))))
        _frame_buffers[camera_id] = buf
    return buf


def _save_clip(camera_id: str, usecase_name: str, buffer: deque, fps: float):
    if not buffer:
        return None
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(EVIDENCE_DIR, f"{usecase_name.lower()}_{camera_id}_{ts}.mp4")
    h, w = buffer[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    try:
        for f in buffer:
            writer.write(f)
    finally:
        writer.release()
    return path


def run_event_logic(batch: dict, usecase_name: str) -> dict:
    """Mandatory entrypoint. The Event orchestrator loads this module and
    calls this function by name; do not rename it."""
    camera_id = batch.get("camera_metadata", {}).get("camera_id", "default")
    frame_metadata = batch.get("frame_metadata", {})
    fps = frame_metadata.get("fps", DEFAULT_FPS)

    frame = frame_metadata.get("frame")
    if isinstance(frame, (bytes, bytearray)):
        frame = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)

    buffer = _get_buffer(camera_id, fps)
    if frame is not None:
        buffer.append(frame.copy())

    bl_results = batch.get("bl_results", {})
    if not bl_results.get("alert"):
        return batch

    if _is_duplicate(camera_id):
        return batch

    num_faces = len(bl_results.get("faces", []))
    logger.warning("[%s] %d usable face(s) found on camera=%s", usecase_name, num_faces, camera_id)

    clip_path = _save_clip(camera_id, usecase_name, buffer, fps)
    if clip_path:
        logger.info("Saved evidence clip to %s", clip_path)

    return batch
