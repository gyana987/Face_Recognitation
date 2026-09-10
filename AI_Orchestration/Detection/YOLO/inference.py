"""Person detector.

Follows the AI_VMS Detection module contract: a class wrapping the model, plus
a module-level run_inference(frames, metadata, device=None) entrypoint the
orchestrator dynamically loads by folder name (Detection/YOLO/inference.py).
Structure/style matches AI_VMS's real Detection/YOLO/inference.py exactly
(YOLOModel class, GPU->CPU fallback, per-device singleton cache) -- the only
addition is `imgsz` support (see module docstring note in README.md: this is
a small, backward-compatible patch to merge into the SHARED real file, not a
new model folder, since person detection there already exists).

imgsz matters here specifically because this usecase needs to catch distant/
small people in the frame: YOLO's default 640px internal resize shrinks far-
away people to near-nothing before detection even runs. Raising imgsz (via
config: PERSON_IMGSZ, default 1280) fixes that at some added inference cost.
"""

import logging
import os
import sys
import threading

# Add parent AI_Orchestration dir to sys.path (same bootstrap AI_VMS's real
# Detection/YOLO/inference.py uses) so this resolves regardless of the
# caller's working directory.
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from constant.constants import Constants
from services.config_reader import cfg

logger = logging.getLogger("YOLOInference")

try:
    import torch
except ImportError:
    torch = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

PERSON_CLASS_ID = 0


class YOLOModel:
    """Person-detection wrapper. Same shape as AI_VMS's real YOLOModel class."""

    def __init__(self, device: str):
        self.device = device
        self.model = None
        self.lock = threading.Lock()
        self.load_model()

    @staticmethod
    def get_weights_path() -> str:
        model_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(model_dir, "yolov8n.pt")

    def load_model(self):
        weights_path = self.get_weights_path()
        if YOLO is None:
            raise ImportError("Ultralytics library is not installed. Please install 'ultralytics'.")

        target_device = self.device
        try:
            if "cuda" in target_device.lower():
                if torch and torch.cuda.is_available():
                    self.model = YOLO(weights_path).to(target_device)
                    logger.info(f"YOLO model loaded successfully on device: {target_device}")
                else:
                    logger.warning(f"CUDA requested ({target_device}) but torch/CUDA is not available. Falling back to CPU.")
                    self.model = YOLO(weights_path).to("cpu")
                    self.device = "cpu"
            else:
                self.model = YOLO(weights_path).to("cpu")
                logger.info("YOLO model loaded successfully on device: cpu")
        except Exception as e:
            logger.warning(
                f"Failed to load YOLO model on GPU device '{target_device}' (likely full or unavailable). "
                f"Falling back to CPU. Error: {e}"
            )
            self.model = YOLO(weights_path).to("cpu")
            self.device = "cpu"

    def run(self, frames: list, metadata: dict) -> list:
        """
        Args:
            frames (list): List of decoded numpy BGR images.
            metadata (dict): optional "confidence" / "imgsz" overrides, and
                "tracking" (bool, default False) -- when True, uses
                model.track() instead of model.predict() so each detection
                carries a persistent "track_id" across frames (needed by
                usecases that pick a single best face per person over time,
                e.g. MISSING_PERSON_FACE_EXTRACTION).

        Returns:
            list: one list of detections per frame:
                  [{"xyxy": [x1,y1,x2,y2], "confidence": 0.93, "class_name": "person",
                    "id": 0, "track_id": 3 or None}, ...]
        """
        if not frames:
            return []

        conf = metadata.get("confidence", cfg.get_float(Constants.PERSON_CONF_KEY, 0.25))
        imgsz = metadata.get("imgsz", cfg.get_int(Constants.PERSON_IMGSZ_KEY, 1280))
        is_tracking = metadata.get("tracking", False)
        tracker = metadata.get("tracker", cfg.get_str(Constants.TRACKER_KEY, "bytetrack.yaml"))

        with self.lock:
            if self.model is None:
                logger.error("Model is not initialized.")
                return [[] for _ in frames]
            try:
                predict_kwargs = dict(classes=[PERSON_CLASS_ID], conf=conf, imgsz=imgsz,
                                       device=self.device, verbose=False)
                if is_tracking:
                    results = self.model.track(frames, persist=True, tracker=tracker, **predict_kwargs)
                else:
                    results = self.model.predict(frames, **predict_kwargs)
            except Exception as e:
                logger.error(f"YOLO inference failed: {e}")
                return [[] for _ in frames]

        batch_detections = []
        for res in results:
            frame_detections = []
            if res.boxes is not None:
                for box in res.boxes:
                    det = {
                        "xyxy": box.xyxy[0].tolist(),
                        "confidence": float(box.conf[0]),
                        "class_name": "person",
                        "id": PERSON_CLASS_ID,
                    }
                    if hasattr(box, "id") and box.id is not None:
                        det["track_id"] = int(box.id[0].item())
                    else:
                        det["track_id"] = None
                    frame_detections.append(det)
            batch_detections.append(frame_detections)
        return batch_detections


_model_instances = {}
_model_init_lock = threading.Lock()


def resolve_device() -> str:
    device_config = cfg.get_str(Constants.DEVICE_KEY, "cpu")
    if device_config.isdigit():
        if torch and torch.cuda.is_available():
            return f"cuda:{device_config}"
        return "cpu"
    return device_config


def run_inference(frames: list, metadata: dict, device: str = None) -> list:
    """Mandatory entrypoint. The orchestrator loads this module and calls this
    function by name; do not rename it.

    metadata may include "tracker_key" (e.g. a camera_id) -- when tracking is
    on, ByteTrack's persist=True state lives INSIDE the ultralytics model
    object, so two different video streams sharing one model instance would
    have their tracker state interleaved frame-by-frame, corrupting track_ids
    for both. Instances are cached per (device, tracker_key) so each camera
    gets its own model + tracker, never sharing tracking state with another
    camera. Callers that don't pass tracker_key (or don't use tracking) all
    share one "default" instance, same as before -- single-camera behavior
    is unchanged."""
    global _model_instances
    if device is None:
        device = resolve_device()
    metadata = metadata or {}
    key = (device, metadata.get("tracker_key", "default"))
    if key not in _model_instances:
        with _model_init_lock:
            if key not in _model_instances:
                _model_instances[key] = YOLOModel(device)
    return _model_instances[key].run(frames, metadata)
